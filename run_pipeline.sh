#!/usr/bin/env bash
#
# ==============================================================================
#  VIPE -> Gaussian Splatting Pipeline
# ==============================================================================
#
#  Takes a video OR a directory of images, runs it through VIPE (depth +
#  trajectory + COLMAP-format camera data), feeds that COLMAP data into
#  Gaussian Splatting for training, renders the trained model, and stitches
#  the renders into an mp4.
#
#  USAGE:
#     ./run_pipeline.sh -i /path/to/input_video.mp4 -n scene_name [options]
#     ./run_pipeline.sh -I /path/to/image_dir -n scene_name [options]
#
#  REQUIRED (use exactly one of -i / -I):
#     -i    Input video file
#     -I    Input image directory (e.g. vipe-main/images/zavod70)
#     -n    Scene name (used to name output folders, e.g. "zavod70")
#
#  OPTIONAL:
#     -v    Path to VIPE repo               (default: ./vipe-main)
#     -g    Path to Gaussian Splatting repo  (default: ./gaussian-splatting)
#     -o    Output/workspace directory       (default: ./pipeline_out)
#     -t    Training iterations              (default: 30000)
#     -f    Render video framerate           (default: 30)
#     -d    VIPE->COLMAP depth_step          (default: 8)
#     --skip-setup     Skip env creation / dependency installation
#     --skip-vipe      Skip VIPE stage (reuse existing colmap_out data)
#     --skip-train     Skip training (reuse existing checkpoint)
#     --clean-env      Remove and recreate the conda envs from scratch
#
#  EXAMPLES:
#     ./run_pipeline.sh -i ~/videos/zavod70.mp4 -n zavod70 -t 30000
#     ./run_pipeline.sh -I ./vipe-main/images/zavod70 -n zavod70
#
#  See README.md for full usage and CHANGELOG.md for the reasoning behind
#  non-obvious parts of this script (CUDA_HOME override, path resolution
#  order, etc).
#
# ==============================================================================

set -eo pipefail
# Not using -u (nounset): breaks conda's own activate.d hooks. See CHANGELOG.md.

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VIPE_DIR="./vipe-main"
GS_DIR="./gaussian-splatting"
OUT_DIR="./pipeline_out"
ITERATIONS=30000
FPS=30
SKIP_SETUP=false
SKIP_VIPE=false
SKIP_TRAIN=false
CLEAN_ENV=false
INPUT_VIDEO=""
INPUT_IMAGE_DIR=""
SCENE_NAME=""

VIPE_CONDA_ENV="cu128"   # conda env: CUDA toolchain + native build deps for VIPE
GS_ENV="gaussian_splatting"

DEPTH_STEP=8              # frame stride used by vipe_to_colmap.py

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i) INPUT_VIDEO="$2"; shift 2 ;;
        -I) INPUT_IMAGE_DIR="$2"; shift 2 ;;
        -n) SCENE_NAME="$2"; shift 2 ;;
        -v) VIPE_DIR="$2"; shift 2 ;;
        -g) GS_DIR="$2"; shift 2 ;;
        -o) OUT_DIR="$2"; shift 2 ;;
        -t) ITERATIONS="$2"; shift 2 ;;
        -f) FPS="$2"; shift 2 ;;
        -d) DEPTH_STEP="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=true; shift ;;
        --skip-vipe) SKIP_VIPE=true; shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --clean-env) CLEAN_ENV=true; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCENE_NAME" ]]; then
    echo "ERROR: -n <scene_name> is required."
    echo "Run with -h for usage."
    exit 1
fi

if [[ -z "$INPUT_VIDEO" && -z "$INPUT_IMAGE_DIR" ]]; then
    echo "ERROR: provide either -i <video> or -I <image_dir>."
    echo "Run with -h for usage."
    exit 1
fi

if [[ -n "$INPUT_VIDEO" && -n "$INPUT_IMAGE_DIR" ]]; then
    echo "ERROR: pass only one of -i <video> or -I <image_dir>, not both."
    exit 1
fi

if [[ -n "$INPUT_VIDEO" && ! -f "$INPUT_VIDEO" ]]; then
    echo "ERROR: Input video not found at: $INPUT_VIDEO"
    exit 1
fi

if [[ -n "$INPUT_IMAGE_DIR" && ! -d "$INPUT_IMAGE_DIR" ]]; then
    echo "ERROR: Input image directory not found at: $INPUT_IMAGE_DIR"
    exit 1
fi

# Resolve to absolute paths now, before any `pushd` later in the script.
# See CHANGELOG.md ("Absolute path resolution") for why this matters.
[[ -n "$INPUT_VIDEO" ]] && INPUT_VIDEO="$(realpath "$INPUT_VIDEO")"
[[ -n "$INPUT_IMAGE_DIR" ]] && INPUT_IMAGE_DIR="$(realpath "$INPUT_IMAGE_DIR")"

COLMAP_OUT="$VIPE_DIR/colmap_out/$SCENE_NAME"
GS_OUTPUT="$GS_DIR/output/$SCENE_NAME"
FINAL_VIDEO="$OUT_DIR/${SCENE_NAME}_demo.mp4"

# Same reasoning — resolved up front so later `pushd` calls can't break them.
COLMAP_OUT="$(realpath -m "$COLMAP_OUT")"
GS_OUTPUT="$(realpath -m "$GS_OUTPUT")"
FINAL_VIDEO="$(realpath -m "$FINAL_VIDEO")"

mkdir -p "$OUT_DIR"

log() { echo -e "\n\033[1;32m==>\033[0m $1"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $1"; }

# ---------------------------------------------------------------------------
# Stage 0: Sanity checks
# ---------------------------------------------------------------------------
log "Stage 0: Checking prerequisites"

command -v conda >/dev/null 2>&1 || { echo "conda not found. Install Anaconda/Miniconda first."; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git not found."; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found. Install with: sudo apt install ffmpeg"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not found. Install with: sudo apt install curl"; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    warn "nvidia-smi not found — GPU may not be available. Training will be very slow or fail."
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# Stage 1: VIPE environment setup
#   VIPE uses conda for the CUDA toolchain/native build deps, and `uv` to
#   manage the actual Python environment in vipe-main/.venv (per its README).
# ---------------------------------------------------------------------------
if [[ "$SKIP_SETUP" == false ]]; then
    log "Stage 1: Setting up VIPE environment (conda: $VIPE_CONDA_ENV, then uv .venv)"

    if [[ ! -d "$VIPE_DIR" ]]; then
        echo "ERROR: VIPE_DIR not found at $VIPE_DIR."
        echo "Clone it first: git clone https://github.com/nv-tlabs/vipe $VIPE_DIR"
        echo "Or pass -v /path/to/vipe-main if it's already cloned elsewhere."
        exit 1
    fi

    if ! command -v uv >/dev/null 2>&1; then
        warn "uv not found — installing it automatically"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # The installer places uv in ~/.local/bin or ~/.cargo/bin depending on version
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        command -v uv >/dev/null 2>&1 || { echo "ERROR: uv install failed. Install manually: https://docs.astral.sh/uv/"; exit 1; }
        log "uv installed: $(uv --version)"
    fi

    # --clean-env: force a fresh env in case an existing one was built
    # before a fix (e.g. the CUDA_HOME fix below) was in place.
    if [[ "$CLEAN_ENV" == true ]] && conda env list | grep -q "^$VIPE_CONDA_ENV "; then
        log "Removing existing conda env '$VIPE_CONDA_ENV' (--clean-env)"
        conda env remove -n "$VIPE_CONDA_ENV" -y
    fi

    if conda env list | grep -q "^$VIPE_CONDA_ENV "; then
        warn "Conda env '$VIPE_CONDA_ENV' already exists, skipping creation."
    else
        pushd "$VIPE_DIR" >/dev/null
        conda env create -f envs/cu128.yml
        popd >/dev/null
    fi

    conda activate "$VIPE_CONDA_ENV"
    pushd "$VIPE_DIR" >/dev/null

    # --clean-env also wipes uv's .venv, since it can carry over stale
    # builds of the CUDA extensions from before a fix was applied.
    if [[ "$CLEAN_ENV" == true && -d .venv ]]; then
        log "Removing existing .venv (--clean-env)"
        rm -rf .venv
    fi

    # Force the conda env's own CUDA toolkit for building (not any system
    # CUDA install). See CHANGELOG.md ("CUDA_HOME / PATH override").
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    log "Using CUDA_HOME=$CUDA_HOME (nvcc: $(command -v nvcc))"

    # Creates/updates .venv and installs the vipe package + its deps
    uv sync

    popd >/dev/null
    conda deactivate
else
    log "Stage 1: Skipping VIPE setup (--skip-setup)"
fi

# ---------------------------------------------------------------------------
# Stage 2: Run VIPE inference, then convert its output to COLMAP format
# ---------------------------------------------------------------------------
if [[ "$SKIP_VIPE" == false ]]; then
    conda activate "$VIPE_CONDA_ENV"
    pushd "$VIPE_DIR" >/dev/null
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CONDA_PREFIX/bin:$PATH"
    # shellcheck disable=SC1091
    source .venv/bin/activate

    # Known fix: PyTorch's cuSOLVER-based lstsq can throw
    # CUSOLVER_STATUS_INVALID_VALUE during depth alignment on some CUDA/
    # driver combinations, even on valid (non-NaN) data. Forcing the MAGMA
    # linalg backend works around it. Patched idempotently so this applies
    # for anyone running the pipeline, not just a manual local edit.
    # See CHANGELOG.md ("MAGMA linalg backend patch") for details.
    ALIGNMENT_FILE="vipe/priors/depth/alignment.py"
    if [[ -f "$ALIGNMENT_FILE" ]] && ! grep -q "preferred_linalg_library" "$ALIGNMENT_FILE"; then
        log "Applying known fix: forcing MAGMA linalg backend in $ALIGNMENT_FILE"
        { echo 'import torch'; echo 'torch.backends.cuda.preferred_linalg_library("magma")'; echo; cat "$ALIGNMENT_FILE"; } > "$ALIGNMENT_FILE.tmp"
        mv "$ALIGNMENT_FILE.tmp" "$ALIGNMENT_FILE"
    fi

    if [[ -n "$INPUT_IMAGE_DIR" ]]; then
        log "Stage 2a: Running 'vipe infer' on image directory $INPUT_IMAGE_DIR"
        vipe infer --image-dir "$INPUT_IMAGE_DIR" -o ./vipe_test_out -p default -v
    else
        log "Stage 2a: Running 'vipe infer' on video $INPUT_VIDEO"
        vipe infer "$INPUT_VIDEO" -o ./vipe_test_out -p default -v
    fi

    log "Stage 2b: Converting VIPE output to COLMAP format (depth_step=$DEPTH_STEP)"
    python3 scripts/vipe_to_colmap.py vipe_test_out \
        --sequence "$SCENE_NAME" \
        --depth_step "$DEPTH_STEP" \
        --output colmap_out

    deactivate
    popd >/dev/null
    conda deactivate

    # Some vipe_to_colmap.py versions write cameras.txt/images.txt/
    # points3D.txt directly into COLMAP_OUT root instead of nesting them
    # under sparse/0/, which is the standard COLMAP layout Gaussian
    # Splatting's train.py expects. Normalize it if needed.
    # See CHANGELOG.md ("COLMAP output layout normalization").
    if [[ ! -d "$COLMAP_OUT/sparse/0" ]] && [[ -f "$COLMAP_OUT/cameras.txt" ]]; then
        log "Normalizing COLMAP output into sparse/0/ layout expected by Gaussian Splatting"
        mkdir -p "$COLMAP_OUT/sparse/0"
        mv "$COLMAP_OUT/cameras.txt" "$COLMAP_OUT/images.txt" "$COLMAP_OUT/points3D.txt" "$COLMAP_OUT/sparse/0/"
    fi

    if [[ ! -d "$COLMAP_OUT/sparse/0" ]]; then
        echo "ERROR: Expected COLMAP output at $COLMAP_OUT/sparse/0 was not produced."
        exit 1
    fi

    # vipe_to_colmap.py was observed writing image filenames in images.txt
    # with a redundant "images/" prefix (e.g. "images/frame_000046.jpg")
    # instead of a bare filename. Gaussian Splatting's loader joins this
    # name onto its own images/ folder path, producing a doubled
    # images/images/frame_000046.jpg that doesn't exist. Strip the prefix
    # if present. See CHANGELOG.md ("Doubled images/ path in images.txt").
    IMAGES_TXT="$COLMAP_OUT/sparse/0/images.txt"
    if [[ -f "$IMAGES_TXT" ]] && grep -q " images/" "$IMAGES_TXT"; then
        log "Stripping redundant 'images/' prefix from filenames in images.txt"
        sed -i 's| images/| |g' "$IMAGES_TXT"
    fi

    # Gaussian Splatting's loader caches a converted points3D.ply the
    # first time it reads points3D.txt, and silently reuses that cache on
    # every later run — even if points3D.txt has since changed (e.g. after
    # regenerating with a different --depth_step). Delete any stale cache
    # so training always reflects the current points3D.txt.
    # See CHANGELOG.md ("Stale cached points3D.ply").
    if [[ -f "$COLMAP_OUT/sparse/0/points3D.ply" ]]; then
        log "Removing stale cached points3D.ply so it regenerates from current points3D.txt"
        rm -f "$COLMAP_OUT/sparse/0/points3D.ply"
    fi
else
    log "Stage 2: Skipping VIPE run (--skip-vipe), reusing existing data at $COLMAP_OUT"
    if [[ ! -d "$COLMAP_OUT/sparse/0" ]]; then
        echo "ERROR: No existing COLMAP data found at $COLMAP_OUT/sparse/0"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3: Gaussian Splatting environment setup
# ---------------------------------------------------------------------------
if [[ "$SKIP_SETUP" == false ]]; then
    log "Stage 3: Setting up Gaussian Splatting environment ($GS_ENV)"

    if [[ ! -d "$GS_DIR" ]]; then
        log "Cloning Gaussian Splatting repo into $GS_DIR"
        git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive "$GS_DIR"
    fi

    # --clean-env: same rationale as the VIPE env above.
    if [[ "$CLEAN_ENV" == true ]] && conda env list | grep -q "^$GS_ENV "; then
        log "Removing existing conda env '$GS_ENV' (--clean-env)"
        conda env remove -n "$GS_ENV" -y
    fi

    if conda env list | grep -q "^$GS_ENV "; then
        warn "Conda env '$GS_ENV' already exists, skipping creation."
    else
        conda create -n "$GS_ENV" python=3.10 -y
    fi

    conda activate "$GS_ENV"
    pushd "$GS_DIR" >/dev/null

    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    pip install "numpy<2" plyfile tqdm opencv-python setuptools==69.5.1

    # Known fix: rasterizer_impl.h uses uint32_t/uint64_t/std::uintptr_t
    # without including <cstdint>/<cstddef>. Older/looser GCC versions
    # pulled these in transitively; newer GCC does not, causing build
    # failures like "identifier uint32_t is undefined". Patched
    # idempotently. See CHANGELOG.md ("Missing cstdint/cstddef includes").
    RASTER_HEADER="submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
    if [[ -f "$RASTER_HEADER" ]] && ! grep -q "#include <cstdint>" "$RASTER_HEADER"; then
        log "Applying known fix: adding missing <cstdint>/<cstddef> includes to $RASTER_HEADER"
        { echo '#include <cstdint>'; echo '#include <cstddef>'; echo; cat "$RASTER_HEADER"; } > "$RASTER_HEADER.tmp"
        mv "$RASTER_HEADER.tmp" "$RASTER_HEADER"
    fi

    # Build CUDA submodules from source
    pushd submodules/diff-gaussian-rasterization >/dev/null
    python setup.py install
    popd >/dev/null

    pushd submodules/simple-knn >/dev/null
    python setup.py install
    popd >/dev/null

    popd >/dev/null
    conda deactivate
else
    log "Stage 3: Skipping Gaussian Splatting setup (--skip-setup)"
fi

# ---------------------------------------------------------------------------
# Stage 4: Train Gaussian Splatting model on VIPE's COLMAP data
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == false ]]; then
    log "Stage 4: Training Gaussian Splatting model ($ITERATIONS iterations)"

    conda activate "$GS_ENV"
    pushd "$GS_DIR" >/dev/null

    python train.py \
        -s "$COLMAP_OUT" \
        -m "$GS_OUTPUT" \
        --iterations "$ITERATIONS" \
        -r 2 \
        --densify_until_iter 5000 \
        --densification_interval 500 \
        --densify_grad_threshold 0.0003

    popd >/dev/null
    conda deactivate
else
    log "Stage 4: Skipping training (--skip-train), reusing checkpoint at $GS_OUTPUT"
fi

# ---------------------------------------------------------------------------
# Stage 5: Render trained model
# ---------------------------------------------------------------------------
log "Stage 5: Rendering trained model"

conda activate "$GS_ENV"
pushd "$GS_DIR" >/dev/null

python render.py -m "$GS_OUTPUT"

popd >/dev/null
conda deactivate

RENDER_DIR="$GS_OUTPUT/train/ours_${ITERATIONS}/renders"
if [[ ! -d "$RENDER_DIR" ]]; then
    echo "ERROR: Expected renders at $RENDER_DIR were not found."
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 6: Stitch renders into a demo video
# ---------------------------------------------------------------------------
log "Stage 6: Building demo video with ffmpeg"

ffmpeg -y -framerate "$FPS" \
    -i "$RENDER_DIR/%05d.png" \
    -c:v libx264 -pix_fmt yuv420p \
    "$FINAL_VIDEO"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "Pipeline complete!"
echo "  Scene:        $SCENE_NAME"
echo "  COLMAP data:  $COLMAP_OUT"
echo "  GS checkpoint: $GS_OUTPUT"
echo "  Renders:      $RENDER_DIR"
echo "  Demo video:   $FINAL_VIDEO"
