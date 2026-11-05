# Native inference deployment

HNDL can compile an eval-mode model into a native PyTorch AOTInductor `.pt2`
package. Loading the package runs the compiled graph without rebuilding or
interpreting the HNDL definition. Export is explicit: ordinary model
construction and `import hndl` do not invoke a compiler.

```python
import torch
from hndl.torch import network
from hndl.deployment import export, load

model = network(
    "linear(128); relu(); linear(10)",
    input_shape=("B", 64),
    output_shape=("B", 10),
    device="cpu",
).eval()
x = torch.randn(8, 64)

artifact = export(model, (x,), "./deployment-cache")
predict = load(artifact)
scores = predict(x)
```

Export uses the generated prepared executor when available. It shares the
original modules while tracing, so the package contains the model's current
weights. It does not randomly initialize a replacement model. Exporting again
after training or loading another checkpoint creates a different artifact.

Named examples are accepted as a mapping, and loaded models accept positional
or named tensors in the original input order. Multiple outputs retain their
named dictionary interface. Single outputs are tensors.

```python
artifact = export(model, {"x": x}, "./deployment-cache")
scores = load(artifact)(x=x)
print(artifact.cache_hit, artifact.path)
```

## Input and runtime contracts

Each artifact specializes one exact input signature: shapes, strides, dtypes,
and devices. A different batch size or layout requires another export. The
loaded Python callable checks these contracts before entering native code.
Input storage offsets may differ when shape and strides agree.

Deployment is inference-only. Call `model.eval()` before export, detach inputs
that require gradients, and continue using the original HNDL model for
training. The loaded callable disables gradient recording itself. Stateful
incremental decoding caches are not part of a deployment artifact.

Compilation needs the C++ toolchain used by PyTorch; CUDA compilation also
needs a compatible CUDA toolchain and GPU. The `.pt2` archive contains native
code, weights, and compiled kernels. The JSON manifest next to it records the
environment that produced it. Load checks the package checksum, PyTorch build,
Python version, CPU capabilities or GPU architecture, and numerical execution
settings before loading native code. Rebuild when these differ. An application
that independently establishes compatibility can explicitly pass
`check_environment=False` to `load`.

Keep the `.pt2` archive and its adjacent `.json` manifest together when copying
them to another directory. `inspect_artifact(path)` reads the manifest without
importing PyTorch or loading executable code. C++ applications can use
[PyTorch's AOTInductor package loader](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html);
they must enforce the same input and environment contracts themselves.

The C++ compiler identity, Triton version, and build environment also
participate in export cache keys. Loading a package does not query the C++
compiler, so the compiler need not be present solely for this compatibility
check.

## Artifact reuse and storage bounds

`export` uses a content-addressed cache beneath the directory you provide. Its
key includes:

- The resolved plan and generated executor implementation.
- Every parameter and buffer, including non-persistent buffers.
- Operator implementation files, forward functions, closures, Python settings,
  forward hooks, and the captured computation and constants.
- Input signatures, compiler options, and the target runtime and hardware.

Derived built-in RoPE tables and attention mask caches do not change the key.
Custom module attributes are fingerprinted; settings that cannot be
fingerprinted produce an explicit error. Do not mutate the source model while
an export is in progress. Model state and settings are checked again after
compilation, and changes discard that build.

Every export captures the graph, including a cache hit. This makes a changed
helper or Python setting visible before reusing a native package; the expensive
native compilation is skipped on a hit.

```python
artifact = export(
    model, (x,), "./deployment-cache",
    max_entries=4,
    max_bytes=512 * 1024**2,
    inductor_configs={"max_autotune": True},
)
```

The defaults retain up to eight packages and two GiB. Least recently used
packages are evicted when necessary. A single package larger than the byte
limit is rejected. `cache=False` forces recompilation of the selected
signature. Package and manifest replacement are protected by a directory lock;
failed compilation leaves no visible partial package.

An evicted artifact must be exported again before it can be loaded. Callables
already loaded remain usable after eviction. These limits cover retained
deployment packages and manifests. Temporary compiler workspace and PyTorch's
separate Inductor cache have their own lifetimes and settings.
