"""Transformer Engine JAX quantization adapters for mixed Linen/NNX models.

The public TE helper ``te_flax.make_dot_general_cls(recipe)`` is the source of
truth for recipe metadata and state. This module only adds:

* layout canonicalization for existing custom projections,
* zero-padding/slicing for TE and cuBLAS alignment constraints, and
* a stateless Linen-to-NNX bridge for forward-only inference.

The adapter intentionally supports recipes without persistent forward state:
Float8CurrentScaling, MXFP8BlockScaling, and an inference-specialized
NVFP4BlockScaling with backward-only RHT/stochastic rounding disabled.
"""

from __future__ import annotations

from collections.abc import Sequence
import functools
import inspect
import math
import os

import jax
import jax.numpy as jnp

GEMM_STATS = {"te": 0, "fallback": 0}
PADDING_STATS = {
    "sites": 0,
    "lhs_elements_added": 0,
    "rhs_elements_added": 0,
}

_DISABLED_ALIASES = {"", "0", "false", "none", "off"}
_RECIPE_ALIASES = {
    "1": "Float8CurrentScaling",
    "true": "Float8CurrentScaling",
    "current": "Float8CurrentScaling",
    "float8currentscaling": "Float8CurrentScaling",
    "mxfp8": "MXFP8BlockScaling",
    "mxfp8blockscaling": "MXFP8BlockScaling",
    "nvfp4": "NVFP4BlockScaling",
    "nvfp4blockscaling": "NVFP4BlockScaling",
}


def selected_recipe_name() -> str | None:
    value = os.environ.get("OPENPI_TE_RECIPE", "")
    normalized = value.strip().lower()
    if normalized in _DISABLED_ALIASES:
        return None
    try:
        return _RECIPE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown OPENPI_TE_RECIPE={value!r}; expected current, mxfp8, or nvfp4") from exc


def env_enabled() -> bool:
    return selected_recipe_name() is not None


def get_recipe(name: str | None = None):
    from transformer_engine.common.recipe import Float8CurrentScaling
    from transformer_engine.common.recipe import Format
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.common.recipe import NVFP4BlockScaling

    name = name or selected_recipe_name()
    if name == "Float8CurrentScaling":
        return Float8CurrentScaling(fp8_format=Format.E4M3)
    if name == "MXFP8BlockScaling":
        return MXFP8BlockScaling(fp8_format=Format.E4M3)
    if name == "NVFP4BlockScaling":
        # These features only affect backward/wgrad. Disabling them makes the
        # recipe stateless for this forward-only accuracy experiment.
        return NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
        )
    raise ValueError("Transformer Engine quantization is disabled")


def _normalize_axes(axes: Sequence[int], ndim: int) -> tuple[int, ...]:
    return tuple(axis if axis >= 0 else ndim + axis for axis in axes)


def _noncontracting_axes(ndim: int, contracting_axes: Sequence[int]) -> tuple[int, ...]:
    contracting = set(contracting_axes)
    return tuple(axis for axis in range(ndim) if axis not in contracting)


def _canonicalize_nn_layout(lhs, rhs, contracting_dims):
    """Move contracting axes to trailing-LHS/leading-RHS order."""
    lhs_contract, rhs_contract = contracting_dims
    lhs_contract = _normalize_axes(lhs_contract, lhs.ndim)
    rhs_contract = _normalize_axes(rhs_contract, rhs.ndim)
    if len(lhs_contract) != len(rhs_contract):
        raise ValueError(f"Mismatched contracting dims: {lhs_contract}, {rhs_contract}")

    lhs_noncontract = _noncontracting_axes(lhs.ndim, lhs_contract)
    rhs_noncontract = _noncontracting_axes(rhs.ndim, rhs_contract)
    lhs_permutation = (*lhs_noncontract, *lhs_contract)
    rhs_permutation = (*rhs_contract, *rhs_noncontract)
    if lhs_permutation != tuple(range(lhs.ndim)):
        lhs = jnp.transpose(lhs, lhs_permutation)
    if rhs_permutation != tuple(range(rhs.ndim)):
        rhs = jnp.transpose(rhs, rhs_permutation)

    lhs_contract = tuple(range(len(lhs_noncontract), lhs.ndim))
    rhs_contract = tuple(range(len(rhs_contract)))
    return lhs, rhs, lhs_contract, rhs_contract


def _alignment_for(scaling_mode) -> int | None:
    from transformer_engine.jax.quantize import ScalingMode

    if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
        return 32
    if scaling_mode.is_nvfp4_scaling:
        return 16
    if scaling_mode.is_tensor_scaling():
        return 16
    return None


def _scale_shape_is_valid(shape, flatten_axis: int, scaling_mode) -> bool:
    try:
        scaling_mode.get_scale_shape_2x(
            tuple(int(dim) for dim in shape),
            is_padded=True,
            flatten_axis=flatten_axis,
            broadcast_2d_scale_shape_to_1d=True,
        )
        return True
    except AssertionError:
        return False


def _pad_axis(value, axis: int, amount: int):
    if amount == 0:
        return value
    padding = [(0, 0)] * value.ndim
    padding[axis] = (0, amount)
    return jnp.pad(value, padding)


def _snap(shape: list[int], axis: int, alignment: int) -> None:
    shape[axis] += (alignment - shape[axis] % alignment) % alignment


def _prepare_aligned_operands(
    lhs,
    rhs,
    lhs_contract,
    rhs_contract,
    *,
    lhs_scaling_mode,
    rhs_scaling_mode,
    alignment: int,
):
    lhs_shape = [int(dim) for dim in lhs.shape]
    rhs_shape = [int(dim) for dim in rhs.shape]
    original_lhs = list(lhs_shape)
    original_rhs = list(rhs_shape)
    lhs_noncontract = _noncontracting_axes(lhs.ndim, lhs_contract)
    rhs_noncontract = _noncontracting_axes(rhs.ndim, rhs_contract)
    lhs_flatten_axis = -len(lhs_contract)
    rhs_flatten_axis = len(rhs_contract) - len(rhs_shape)

    # NVFP4 uses 16-value blocks but cuBLAS requires K to be a multiple of 32.
    k_alignment = 32 if lhs_scaling_mode.is_nvfp4_scaling else alignment
    if rhs_noncontract:
        _snap(rhs_shape, rhs_noncontract[-1], alignment)
    if lhs_contract:
        _snap(lhs_shape, lhs_contract[-1], k_alignment)
        rhs_shape[rhs_contract[-1]] = lhs_shape[lhs_contract[-1]]
    if lhs_noncontract:
        _snap(lhs_shape, lhs_noncontract[-1], alignment)

    for _ in range(256):
        lhs_valid = _scale_shape_is_valid(
            lhs_shape,
            lhs_flatten_axis,
            lhs_scaling_mode,
        )
        rhs_valid = _scale_shape_is_valid(
            rhs_shape,
            rhs_flatten_axis,
            rhs_scaling_mode,
        )
        if lhs_valid and rhs_valid:
            break
        if not rhs_valid and rhs_noncontract:
            rhs_shape[rhs_noncontract[-1]] += alignment
        elif not rhs_valid and rhs_contract:
            rhs_shape[rhs_contract[-1]] += k_alignment
            lhs_shape[lhs_contract[-1]] += k_alignment
        elif not lhs_valid and lhs_noncontract:
            lhs_shape[lhs_noncontract[-1]] += alignment
        elif not lhs_valid and lhs_contract:
            lhs_shape[lhs_contract[-1]] += k_alignment
            rhs_shape[rhs_contract[-1]] += k_alignment
        else:
            raise ValueError(f"Cannot align TE operands: lhs={lhs_shape}, rhs={rhs_shape}")
    else:
        raise ValueError(f"TE operand alignment did not converge: lhs={lhs_shape}, rhs={rhs_shape}")

    lhs_added = math.prod(lhs_shape) - math.prod(original_lhs)
    rhs_added = math.prod(rhs_shape) - math.prod(original_rhs)
    if lhs_added or rhs_added:
        PADDING_STATS["sites"] += 1
        PADDING_STATS["lhs_elements_added"] += lhs_added
        PADDING_STATS["rhs_elements_added"] += rhs_added

    output_slices: list[tuple[int, int]] = []
    for axis, original, aligned in zip(
        range(lhs.ndim),
        original_lhs,
        lhs_shape,
        strict=True,
    ):
        lhs = _pad_axis(lhs, axis, aligned - original)
        if axis in lhs_noncontract and aligned != original:
            output_slices.append((lhs_noncontract.index(axis), original))
    for axis, original, aligned in zip(
        range(rhs.ndim),
        original_rhs,
        rhs_shape,
        strict=True,
    ):
        rhs = _pad_axis(rhs, axis, aligned - original)
        if axis in rhs_noncontract and aligned != original:
            output_slices.append((len(lhs_noncontract) + rhs_noncontract.index(axis), original))
    return lhs, rhs, output_slices


def _slice_output(output, slices: Sequence[tuple[int, int]]):
    for axis, original_size in slices:
        output = jax.lax.slice_in_dim(output, 0, original_size, axis=axis)
    return output


def reset_stats() -> None:
    for stats in (GEMM_STATS, PADDING_STATS):
        for key in stats:
            stats[key] = 0


def make_linen_dot_general_cls(recipe=None, *, strict: bool = True):
    """Return a Linen dot_general module class backed by TE's public helper."""
    import flax.linen as nn
    import transformer_engine.jax.flax as te_flax
    from transformer_engine.jax.quantize import TensorSource
    from transformer_engine.jax.quantize import get_quantize_config_with_recipe
    from transformer_engine.jax.quantize import is_quantize_recipe_supported

    recipe = recipe or get_recipe()
    recipe_name = type(recipe).__name__
    supported, reason = is_quantize_recipe_supported(recipe_name)
    if not supported:
        raise RuntimeError(f"{recipe_name} is not supported: {reason}")

    config = get_quantize_config_with_recipe(recipe)
    lhs_scaling_mode = config.get_scaling_mode(TensorSource.X)
    rhs_scaling_mode = config.get_scaling_mode(TensorSource.KERNEL)
    alignment = _alignment_for(lhs_scaling_mode)
    official_dot_cls = te_flax.make_dot_general_cls(recipe)

    class AlignedTEDotGeneral(nn.Module):
        @nn.compact
        def __call__(
            self,
            lhs,
            rhs,
            dimension_numbers,
            precision=None,
            preferred_element_type=None,
            **kwargs,
        ):
            contracting_dims, batch_dims = dimension_numbers
            if batch_dims != ((), ()):
                GEMM_STATS["fallback"] += 1
                if strict:
                    raise ValueError(f"TE adapter requires empty batch dimensions; got {dimension_numbers}")
                fallback_kwargs = dict(kwargs)
                if precision is not None:
                    fallback_kwargs["precision"] = precision
                if preferred_element_type is not None:
                    fallback_kwargs["preferred_element_type"] = preferred_element_type
                return jax.lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers,
                    **fallback_kwargs,
                )

            lhs, rhs, lhs_contract, rhs_contract = _canonicalize_nn_layout(
                lhs,
                rhs,
                contracting_dims,
            )
            if alignment is not None:
                lhs, rhs, output_slices = _prepare_aligned_operands(
                    lhs,
                    rhs,
                    lhs_contract,
                    rhs_contract,
                    lhs_scaling_mode=lhs_scaling_mode,
                    rhs_scaling_mode=rhs_scaling_mode,
                    alignment=alignment,
                )
            else:
                output_slices = ()

            GEMM_STATS["te"] += 1
            output = official_dot_cls(name="te_dot")(
                lhs,
                rhs,
                ((lhs_contract, rhs_contract), ((), ())),
            )
            return _slice_output(output, output_slices)

    AlignedTEDotGeneral.__name__ = f"AlignedTE_{recipe_name}"
    return AlignedTEDotGeneral


def make_nnx_dot_general(recipe=None, *, strict: bool = True):
    """Bridge the stateless Linen adapter into an NNX-compatible callable."""
    from flax import nnx
    import flax.nnx.bridge as nnx_bridge

    recipe = recipe or get_recipe()
    linen_dot_cls = make_linen_dot_general_cls(recipe, strict=strict)
    te_dot = nnx_bridge.ToNNX(linen_dot_cls())
    dims = (((1,), (0,)), ((), ()))
    te_dot.lazy_init(
        jnp.zeros((128, 128), dtype=jnp.bfloat16),
        jnp.zeros((128, 128), dtype=jnp.bfloat16),
        dims,
    )
    _, dot_state = nnx.split(te_dot)
    if jax.tree_util.tree_leaves(dot_state):
        raise ValueError(
            f"{type(recipe).__name__} created persistent state; this forward-only NNX bridge supports stateless recipes"
        )

    # NNX Linear stores dot_general as a static attribute, so keep the graph
    # node behind a hashable function.
    def dot_general(*args, **kwargs):
        return te_dot(*args, **kwargs)

    return dot_general


def _patch_nnx_linears(model, dot_general) -> int:
    from flax import nnx

    patched = 0
    for _, module in model.iter_modules():
        if isinstance(module, nnx.Linear | nnx.LinearGeneral):
            module.dot_general = dot_general
            if hasattr(module, "dot_general_cls"):
                module.dot_general_cls = None
            patched += 1
    return patched


def module_jit_quantized(method, *jit_args, **jit_kwargs):
    """Freeze an NNX method and install TE only on its ephemeral trace module."""
    import flax.nnx as nnx

    if not (inspect.ismethod(method) and isinstance(method.__self__, nnx.Module)):
        raise ValueError("module_jit_quantized expects a bound NNX module method")

    recipe = get_recipe()
    dot_general = make_nnx_dot_general(recipe)
    graphdef, state = nnx.split(method.__self__)
    reset_stats()

    def function(state: nnx.State, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        _patch_nnx_linears(module, dot_general)
        return method.__func__(module, *args, **kwargs)

    jitted = jax.jit(function, *jit_args, **jit_kwargs)

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        return jitted(state, *args, **kwargs)

    return wrapper
