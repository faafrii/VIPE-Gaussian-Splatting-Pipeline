# Changelog — `run_pipeline.sh`

Notes on why the script is built the way it is. Kept separate from the
script itself so the script stays readable; this is the "why" behind
non-obvious decisions and fixes.

## VIPE toolchain corrected to match actual usage
VIPE's real setup process (per its own README) uses **conda** for the
CUDA toolchain/native build deps, and **`uv`** to manage the actual Python
environment in `vipe-main/.venv`:
```bash
conda env create -f envs/cu128.yml
conda activate cu128
uv sync
```
Stage 2 calls the real `vipe infer` CLI and the real
`scripts/vipe_to_colmap.py` conversion script, with the exact flags
confirmed from a working manual run.

## `-I` flag for image directory input
Added to support passing an image directory (e.g. `images/zavod70`)
directly, as an alternative to `-i <video>`, since `vipe infer` supports
both `--image-dir` and a positional video path.

## `uv` auto-install
If `uv` isn't found, the script installs it via the official installer
(`https://astral.sh/uv/install.sh`) rather than erroring out and requiring
a manual step.

## Dropped `-u` (nounset) from `set -euo pipefail`
Conda's own `activate.d` hooks (e.g. the `cuda-nvcc` package's) reference
variables like `NVCC_PREPEND_FLAGS` without checking if they're set first.
This crashes immediately under strict `nounset` mode. `-e` and `pipefail`
are kept so real errors still stop the script — only `-u` was dropped.

## `CUDA_HOME` / `PATH` override before building CUDA extensions
A system-wide CUDA install (e.g. `/usr/local/cuda-12.2`) can take priority
over the conda env's own CUDA toolkit if it appears earlier in `PATH`.
This caused `nvcc`/`gcc` version-mismatch build failures even though the
conda env itself was correctly isolated. Fixed by explicitly setting:
```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
```
right before any build step, so the env's own `nvcc` always wins.

## `--clean-env` flag
Removes and recreates the conda envs (and `vipe-main/.venv`) from scratch.
Useful when an existing env was built before a fix above existed, and may
still carry over a stale/broken build.

## Absolute path resolution before any `pushd`
Two related bugs, same root cause: paths were being resolved with
`realpath` *after* the script had already `pushd`'d into a different
working directory, so they resolved relative to the wrong location.

- `-i`/`-I` input paths: broke when a relative path like
  `./vipe-main/images/zavod70` was resolved after already `cd`-ing into
  `vipe-main/` for the VIPE stage.
- `COLMAP_OUT` / `GS_OUTPUT` / `FINAL_VIDEO`: broke when resolved inside a
  `pushd` into `gaussian-splatting/` in Stage 4/5, producing a doubled
  `gaussian-splatting/gaussian-splatting/output/...` path. This caused
  `train.py` to receive an invalid `-s`/`-m` and fail with
  `AssertionError: Could not recognize scene type!`.

Fix: all input/output paths are now resolved to absolute paths once, near
the top of the script, before any `pushd` occurs anywhere.

## Documented where VIPE comes from
The script assumed `vipe-main/` already existed locally but never
documented how to get it — a checker cloning just this repo had no way to
obtain VIPE. Unlike Gaussian Splatting, VIPE can't be auto-cloned by the
script because the user's dataset must be manually placed inside it
(`vipe-main/images/<scene>/`) before running, so cloning it is
unavoidably a manual first step either way. README now documents the
exact clone command (`git clone https://github.com/nv-tlabs/vipe
vipe-main`) and dataset placement steps; the script's error message also
now suggests the clone command if `vipe-main/` is missing.

## MAGMA linalg backend patch
VIPE's depth alignment step (`vipe/priors/depth/alignment.py`) calls
`torch.linalg.lstsq`, which by default routes through PyTorch's cuSOLVER
backend on GPU. On some CUDA/driver combinations, this throws:
```
torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE,
when calling `cusolverDnSormqr_bufferSize(...)`
```
This happened consistently on a valid, non-corrupted 126-image dataset
that had previously worked, and reproduced identically on a completely
fresh clone/environment — ruling out bad input data or a one-off
environment issue. It's a cuSOLVER-specific edge case, not a NaN-in-data
problem despite what the error message suggests.

**Fix:** force PyTorch to use its MAGMA backend for linear algebra instead
of cuSOLVER:
```python
import torch
torch.backends.cuda.preferred_linalg_library("magma")
```
This routes the same computation through different underlying code and
avoids the buggy cuSOLVER path entirely.

Since this needs to live inside VIPE's own source file (a third-party
dependency not tracked in this repo), `run_pipeline.sh` applies it
automatically and idempotently at the start of Stage 2, right before
`vipe infer` runs — so it works for anyone running the pipeline fresh,
not just as a manual local edit.

## COLMAP output layout normalization
`scripts/vipe_to_colmap.py` (part of the VIPE repo) was observed writing
`cameras.txt` / `images.txt` / `points3D.txt` directly into
`colmap_out/<scene>/`, rather than nesting them under
`colmap_out/<scene>/sparse/0/` — the standard COLMAP layout that Gaussian
Splatting's `train.py` requires. This appears to be another effect of
VIPE's ongoing updates (same category as the MAGMA patch above): a fresh
clone can behave differently than an older commit that previously worked,
even though the conversion itself completes successfully and logs no
error. The script now detects this case and moves the three files into
`sparse/0/` automatically before checking for/handing off to Gaussian
Splatting.

## Missing cstdint/cstddef includes in diff-gaussian-rasterization
Building the `diff-gaussian-rasterization` CUDA submodule failed with:
```
cuda_rasterizer/rasterizer_impl.h(24): error: namespace "std" has no
member "uintptr_t"
cuda_rasterizer/rasterizer_impl.h(40): error: identifier "uint32_t" is
undefined
```
`rasterizer_impl.h` uses `uint32_t`, `uint64_t`, and `std::uintptr_t`
without including `<cstdint>` (for the `uint*_t` types) or `<cstddef>`
(for `std::size_t`). Older/looser GCC versions pulled these in
transitively through other standard headers; newer GCC does not,
so the build fails outright with this compiler. This is a known upstream
issue in the `graphdeco-inria/gaussian-splatting` repo itself, not
specific to this pipeline. Since it's third-party code not tracked in
this repo, `run_pipeline.sh` patches the two missing includes in
automatically and idempotently in Stage 3, right before building the
submodule.

## Doubled images/ path in images.txt
Training failed with:
```
FileNotFoundError: [Errno 2] No such file or directory:
'.../colmap_out/zavod70/images/images/frame_000046.jpg'
```
`vipe_to_colmap.py` wrote image filenames into `images.txt` with a
redundant `images/` prefix already included (e.g.
`images/frame_000046.jpg`), rather than the bare filename that standard
COLMAP format expects (just `frame_000046.jpg`, relative to the
`images/` folder). Gaussian Splatting's loader joins the name from
`images.txt` onto its own `images/` folder path
(`camera_utils.py::loadCam`), so a name that already includes `images/`
produces a doubled `images/images/...` path that doesn't exist. Same
category as the sparse/0 layout issue above — a newer `vipe_to_colmap.py`
not quite matching the classic COLMAP convention. Fixed by stripping the
redundant prefix from `images.txt` right after the sparse/0
normalization step.

## Stale cached points3D.ply
Training kept hitting the same `CUDA out of memory` error, with the same
allocation size and even the same loss value, even after regenerating
`points3D.txt` with a much larger `--depth_step` (fewer points).
Root cause: Gaussian Splatting's data loader
(`scene/dataset_readers.py::readColmapSceneInfo`) converts
`points3D.txt`/`.bin` into a `points3D.ply` file the *first* time it
reads a given source folder, and caches it there
(`sparse/0/points3D.ply`). On every later run, if that `.ply` already
exists, it's loaded directly and `points3D.txt` is never re-read — so a
regenerated, smaller `points3D.txt` was being silently ignored in favor
of the original 12-million-point cache from the very first training
attempt. Fixed by deleting any stale `points3D.ply` right after the
COLMAP layout/path fixes, so training always reflects the current
`points3D.txt`. Note this only runs automatically when Stage 2 executes;
a `--skip-vipe` re-run that reuses old COLMAP data needs the stale
`.ply` removed manually if `points3D.txt` was regenerated out-of-band.

## Point cloud size cap
Training repeatedly hit `CUDA out of memory` with millions of initial
Gaussians (observed: ~12M points at default `depth_step=8`, ~3M at
`depth_step=32`), across multiple separate clones/folders. Increasing
`depth_step` further wasn't a reliable fix: since it behaves as a frame
stride (dropping whole frames' depth maps rather than sub-sampling pixels
within a frame), pushing it high enough to meaningfully shrink the point
count also drops too many frames and starts hurting reconstruction
coverage. The underlying issue — dense per-pixel depth unprojection
producing millions of points regardless of exact settings — also isn't
guaranteed to behave consistently across different VIPE versions/clones.

Instead of continuing to tune `depth_step` per-environment, the script
now applies a deterministic hard cap (`-m`, default `500000`) directly on
`points3D.txt` after conversion: if the point count exceeds the cap, a
uniform random sample of that size is kept. This is safe for Gaussian
Splatting specifically because its stock `train.py`/`scene` loader only
uses `points3D.txt` for initial Gaussian positions/colors — the per-image
2D-to-3D track associations in `images.txt` are read but not validated
against `points3D.txt`, so removing points doesn't break anything else in
the pipeline. Applied automatically, right before the stale-`.ply` cache
cleanup above (so the cache always regenerates from the capped file).
