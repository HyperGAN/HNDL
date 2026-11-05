"""Optional released-ParticleGAN integration; HNDL does not depend on PG."""

from importlib.metadata import PackageNotFoundError, version

import pytest
import torch

from examples.particle_gan_training import make_trainer, real_batch

try:
    pg_version = version("particlegan")
except PackageNotFoundError:
    pytest.skip("optional particlegan==0.9.4 integration", allow_module_level=True)
if pg_version != "0.9.4":
    pytest.skip("integration qualifies particlegan==0.9.4", allow_module_level=True)

def test_released_default_updates_hndl_networks_and_real_particle_prior(monkeypatch):
    def no_cuda(*args, **kwargs):
        pytest.fail("CPU fixture attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    trainer = make_trainer()
    recipe = trainer.recipe
    assert recipe.name == "bispf_optimized"
    assert recipe.prior_regularizer == "sbnp"
    before = [{key: value.detach().clone() for key, value in module.state_dict().items()}
              for module in (trainer.G, trainer.D, trainer.prior)]
    frozen_base = trainer.G["base"].weight.detach().clone()
    assert not trainer.G["base"].weight.requires_grad
    assert torch.count_nonzero(trainer.G["lora_up"].weight) == 0
    for _ in range(recipe.total_steps):
        result = trainer.step(real_batch())
        assert all(torch.isfinite(value).all() for value in result.values()
                   if isinstance(value, torch.Tensor))
    assert result["step"] == 3
    assert result["prior_regularization"] > 0
    for original, module in zip(before, (trainer.G, trainer.D, trainer.prior)):
        assert any(not torch.equal(value, module.state_dict()[key])
                   for key, value in original.items())
    assert torch.equal(trainer.G["base"].weight, frozen_base)
    assert trainer.G["base"].weight.grad is None
    assert torch.count_nonzero(trainer.G["lora_up"].weight) > 0
    assert trainer.sample(5).shape == (5, 4)


def test_hndl_networks_resume_public_trainer_state_exactly():
    trainer = make_trainer()
    trainer.step(real_batch())
    checkpoint = trainer.state_dict()
    expected = trainer.step(real_batch())
    expected_state = trainer.state_dict()
    restored = make_trainer()
    restored.load_state_dict(checkpoint)
    actual = restored.step(real_batch())
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value
    for name, state in expected_state["models"].items():
        for key, value in state.items():
            assert torch.equal(restored.state_dict()["models"][name][key], value)
