# ParticleGAN and particle sliders

HNDL builds the network graphs used by a particle GAN or particle LoRA. The
training application owns attachment to the frozen base, conditioning, particle
bank, objective, optimizer state, sampling and exports. Install the optional
integration with:

```sh
python -m pip install 'hndl[particlegan]'
```

This extra requires `particlegan>=0.9.4,<0.10`; ordinary HNDL installations do
not acquire a ParticleGAN dependency. ParticleGAN 0.9.4 is the qualified release
for the example below. A future compatible release still needs its own check.

## A runnable current-recipe example

From this checkout, with the extra installed:

```sh
CUDA_VISIBLE_DEVICES='' python examples/particle_gan_training.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_particlegan_integration.py
```

The example builds a frozen 4-by-4 base projection and a rank-2 linear update
using native HNDL `linear` and `add` operations. The up projection starts at
zero, so the initial output equals the base output. A separate HNDL MLP is the
critic. ParticleGAN's public `get_recipe()` and `GANTrainer` own the complete
update; the example changes only the synthetic dimensions and update budget:

```python
from particlegan import GANTrainer, get_recipe

recipe = get_recipe(num_particles=32, z_dim=4, batch_size=8, total_steps=3)
trainer = GANTrainer(recipe, generator, critic, seed=23)
metrics = trainer.step(real_batch())
```

At [ParticleGAN 0.9.4](https://github.com/255BITS/ParticleGAN/releases/tag/v0.9.4),
the selected default is `bispf_optimized`: non-saturating logistic losses, the
Born–Infeld critic penalty, SBNP prior regularization, zero-momentum DualNorm
and constant effective learning rates. `GANTrainer` constructs a real learned
particle prior, separate from the low-rank network weights. The regression
checks that both graphs and the prior change, the frozen base stays identical,
and public trainer checkpoint restoration reproduces the next update exactly.

This is a three-update CPU interoperability fixture. It is not a trained image,
audio or language control and does not load a pretrained model. The example's
small ordinary low-rank update also differs from the routed nonlinear adapter
described below.

## Existing routed particle LoRAs

The local P-H3 HNDL integration under review already supplies native HNDL graphs
for a routed particle adapter and a global-mix critic. It is a separate product
package, not an installable HNDL feature. Its audited architecture is:

| Part | Implementation |
| --- | --- |
| Low-rank projection | Rank-8 down/up maps; the up projection starts at zero |
| Shared particle bank | One trainable 128-by-4 tensor owned by the adapter |
| Routing | Width-16 router, scaled dot product, softmax and weighted bank lookup |
| Nonlinear branch | Concatenated input/bank response through a width-48 MLP |
| Attachment | Explicit native linear targets on a frozen host |
| Inference | Exact zero-strength bypass, owned-hook cleanup, checked adapter export/load |
| Critic | Full-response global mix with explicit attention operations supporting double backward |

These graphs compose existing HNDL operations; HNDL does not need to copy the
product's adapter lifecycle or create a second particle bank. The product's
source-bound package pins HNDL 0.10.0 and the Python source closure at
`df5af35d3b4017e1164b011d12264a01546a3ff4`. Network output and input/particle
gradient parity, critic second derivatives, frozen-base ownership and adapter
export/reload are covered by its CPU tests. Preserve that exact environment
for existing artifacts; a dependency update is a new qualification, not an
automatic checkpoint migration.

The local integration remains subject to its product owner's publication and
native qualification process. This guide does not publish its source, artifacts
or weights.

The [HyperGAN integration guide](https://github.com/HyperGAN/HyperGAN/blob/develop/docs/particle-sliders.md)
describes how its trainer consumes these caller-owned networks and particle
parameters. [particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/main/packages/particle-sliders-core)
owns reusable product sessions and response adapters; product repositories own
the pretrained backbone, native projection names, media sampling and controls.

## Version, recipe and trained weights are separate

Installing ParticleGAN 0.9.4 does not make every existing experiment use its
current default. The existing P-H3 HyperGAN cohort explicitly selects `ka2`,
with relativistic loss, VICReg and a cosine schedule. Its routed/gmix network
can run under that cohort without becoming a `bispf_optimized` result. Earlier
published particle-slider collections likewise include historical gmix and
ordinary LoRA objectives. Keep their recorded recipe and checkpoint lineage.

Use the current-recipe example above to check HNDL/ParticleGAN compatibility.
Use the product's own native tests and evaluation to qualify a real media
control. A CPU gradient or checkpoint test establishes neither pretrained
quality nor a newly trained or published slider.

HNDL's `latent_attention` operator uses low-rank attention compression; it is
not the routed particle adapter in this guide. Similarly,
`allow_state_mutation()` and `rebind_state()` permit explicit external adapter
injection but do not train or export a slider by themselves.
