from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import te_quantization


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("current", "Float8CurrentScaling"),
        ("Float8CurrentScaling", "Float8CurrentScaling"),
        ("mxfp8", "MXFP8BlockScaling"),
        ("MXFP8BlockScaling", "MXFP8BlockScaling"),
        ("nvfp4", "NVFP4BlockScaling"),
        ("NVFP4BlockScaling", "NVFP4BlockScaling"),
    ],
)
def test_recipe_selection(monkeypatch, value, expected):
    monkeypatch.setenv("OPENPI_TE_RECIPE", value)
    assert te_quantization.selected_recipe_name() == expected


def test_invalid_recipe(monkeypatch):
    monkeypatch.setenv("OPENPI_TE_RECIPE", "fp7")
    with pytest.raises(ValueError, match="Unknown OPENPI_TE_RECIPE"):
        te_quantization.selected_recipe_name()


@pytest.mark.manual
@pytest.mark.parametrize(
    ("recipe", "marker"),
    [
        ("current", "f8_e4m3"),
        ("mxfp8", "MXFP8"),
        ("nvfp4", "f4_e2m1"),
    ],
)
def test_nnx_bridge_uses_selected_recipe(monkeypatch, recipe, marker):
    pytest.importorskip("transformer_engine.jax")
    monkeypatch.setenv("OPENPI_TE_RECIPE", recipe)
    dot_general = te_quantization.make_nnx_dot_general()
    linear = nnx.Linear(
        128,
        256,
        use_bias=False,
        dtype=jnp.bfloat16,
        dot_general=dot_general,
        rngs=nnx.Rngs(0),
    )
    graphdef, state = nnx.split(linear)

    @jax.jit
    def apply(state, inputs):
        return nnx.merge(graphdef, state)(inputs)

    inputs = jnp.ones((32, 128), dtype=jnp.bfloat16)
    output = apply(state, inputs)
    jax.block_until_ready(output)
    assert output.shape == (32, 256)
    assert marker.lower() in str(jax.make_jaxpr(apply)(state, inputs)).lower()
