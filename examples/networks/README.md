# Authored networks

Each `<name>.hndl` file is a complete HNDL configuration; the sidecar
`<name>.json` gives its contract (`input_shape`, `output_shape`, optional
`input_dtype` and `dtype`), a title and description, an optional expected
`parameters` count, and a `reference` sentence. `input_shape` and
`output_shape` are either one shape array or an object of named shapes, as
`conditional_discriminator.json` shows, and `input_dtype` may likewise name
one dtype per input. `tests/test_network_examples.py`
resolves, builds, and runs every pair on CPU and CUDA, and
`python -m hndl.docs` renders them into `docs/networks.md`.
