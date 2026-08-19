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

## Enable quantized policy inference

```bash
OPENPI_TE_RECIPE=current uv run scripts/serve_policy.py --env LIBERO
OPENPI_TE_RECIPE=mxfp8 uv run scripts/serve_policy.py --env LIBERO
OPENPI_TE_RECIPE=nvfp4 uv run scripts/serve_policy.py --env LIBERO
```

This is intended as a recipe-correctness and accuracy reference. TE JAX does
not currently pre-pack constant weights for inference, so this path should not
be treated as an optimized serving implementation.

## Standalone NNX example

`nnx_custom_projection.py` demonstrates the same bridge with a generic custom
projection and neutral dimensions:

```bash
OPENPI_TE_RECIPE=mxfp8 \
python examples/te_quantization/nnx_custom_projection.py
```

Attention QK/PV products, normalization, embeddings, and convolution remain at
their original precision in this example.
