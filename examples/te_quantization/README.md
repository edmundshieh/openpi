# Transformer Engine JAX quantization example

This branch demonstrates how to use Transformer Engine's public
`te_flax.make_dot_general_cls(recipe)` helper with OpenPI's mixed Linen/NNX
model while preserving the existing parameter tree.

The integration supports:

- `Float8CurrentScaling` with E4M3 forward operands
- `MXFP8BlockScaling`
- forward-only `NVFP4BlockScaling` with backward-only RHT/SR disabled

Standard Linen Dense and attention projections receive a TE
`dot_general_cls`. Existing custom Einsum/dot projections route through the
same class. Native NNX linears use the stateless Linen wrapper through
`nnx.bridge.ToNNX`.

The adapter also canonicalizes non-standard contraction layouts and pads
operands with zeros when TE/cuBLAS alignment requires it. Since these recipes
use amax-based scaling, zero-padding does not alter scale selection.

## Installation

Install a JAX-enabled Transformer Engine build separately. Transformer Engine
is imported lazily and remains optional when `OPENPI_TE_RECIPE` is unset.

## Entry point 1: public OpenPI model integration

This is the end-to-end integration in the public model:

1. [`pi0.py`](../../src/openpi/models/pi0.py) creates the selected Linen
   `dot_general_cls` and passes it into Gemma and SigLIP.
2. [`gemma.py`](../../src/openpi/models/gemma.py),
   [`lora.py`](../../src/openpi/models/lora.py), and
   [`siglip.py`](../../src/openpi/models/siglip.py) route their existing Dense,
   attention-projection, Einsum, and dot operations through that class.
3. [`policy.py`](../../src/openpi/policies/policy.py) installs the same recipe
   on native NNX Linear modules while constructing the jitted inference
   function.
4. [`te_quantization.py`](../../src/openpi/models/te_quantization.py) contains
   the shared recipe, state, layout, padding, and Linen/NNX bridge logic.

Enable this path through `OPENPI_TE_RECIPE`:

```bash
OPENPI_TE_RECIPE=current uv run scripts/serve_policy.py --env LIBERO
OPENPI_TE_RECIPE=mxfp8 uv run scripts/serve_policy.py --env LIBERO
OPENPI_TE_RECIPE=nvfp4 uv run scripts/serve_policy.py --env LIBERO
```

This is intended as a recipe-correctness and accuracy reference. TE JAX does
not currently pre-pack constant weights for inference, so this path should not
be treated as an optimized serving implementation.

## Entry point 2: architecture-neutral NNX example

[`nnx_custom_projection.py`](nnx_custom_projection.py) is a focused reproducer
for the NNX side of the same adapter. It includes:

- a native `nnx.Linear`,
- a custom higher-rank projection with multiple output axes, and
- a custom projection that contracts two axes.

All dimensions and names are intentionally architecture-neutral.

```bash
OPENPI_TE_RECIPE=mxfp8 \
python examples/te_quantization/nnx_custom_projection.py
```

This entry point is not another model implementation. It isolates the
projection patterns needed to review the NNX bridge without requiring model
weights, data transforms, or the rest of OpenPI.

Attention QK/PV products, normalization, embeddings, and convolution remain at
their original precision in this example.
