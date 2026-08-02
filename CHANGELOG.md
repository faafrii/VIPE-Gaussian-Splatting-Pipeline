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
