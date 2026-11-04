"""Exercise HNDL networks with the public ParticleGAN 0.9.4 trainer on CPU.

Install the optional integration: ``pip install 'hndl[particlegan]'``.
The generator is a frozen linear base plus a zero-initialized rank-2 update.
This small synthetic GAN checks interoperability, not pretrained slider quality.
"""

import json

import torch

from hndl.torch import network


def make_trainer():
    """Use the released default objective; override only fixture sizes/budget."""
    from particlegan import GANTrainer, get_recipe

    recipe = get_recipe(num_particles=32, z_dim=4, batch_size=8, total_steps=3)
    generator = network(
        '''
        base = linear(value, 4, bias=False, name="base", trainable=False)
        down = linear(value, 2, bias=False, name="lora_down")
        delta = linear(down, 4, bias=False, name="lora_up", init={"weight": 0})
        add(base, delta)
        ''',
        input_shape={"value": ("B", recipe.z_dim)}, output_shape=("B", 4),
        device="cpu", initialization_seed=11,
    )
    critic = network(
        'linear(16)\nsilu()\nlinear()',
        input_shape=("B", 4), output_shape=("B", 1),
        device="cpu", initialization_seed=17,
    )
    return GANTrainer(recipe, generator, critic, seed=23)


def real_batch():
    """A fixed synthetic batch, with no dataset, model download or GPU access."""
    return torch.linspace(-1, 1, 32).reshape(8, 4)


def main():
    trainer = make_trainer()
    for _ in range(trainer.recipe.total_steps):
        result = trainer.step(real_batch())
        print(json.dumps({key: float(value) if isinstance(value, torch.Tensor) else value
                          for key, value in result.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
