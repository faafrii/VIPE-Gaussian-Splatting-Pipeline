# VIPE → Gaussian Splatting Pipeline

Reconstructs a 3D Gaussian Splatting scene from a video or image sequence.

**Pipeline:** input video/images → **VIPE** (depth + camera pose
estimation) → **COLMAP-format conversion** → **Gaussian Splatting**
(training + rendering) → demo video.

## Quickstart

```bash
git clone https://github.com/faafrii/VIPE-Gaussian-Splatting-Pipeline.git
cd VIPE-Gaussian-Splatting-Pipeline
git clone https://github.com/nv-tlabs/vipe vipe-main

mkdir -p vipe-main/images/your_scene
cp /path/to/your/photos/*.jpg vipe-main/images/your_scene/

chmod +x run_pipeline.sh
./run_pipeline.sh -I ./vipe-main/images/your_scene -n your_scene
```
First run installs everything (conda envs, CUDA extensions) and can take a
while. Output: `pipeline_out/your_scene_demo.mp4`. See below for full
options.

## Files

| File | Purpose |
|---|---|
| `run_pipeline.sh` | Main pipeline script — environment setup through final render |
| `CHANGELOG.md` | Why the script is built the way it is — bug fixes and design decisions |
| `.gitignore` | Excludes venvs, model checkpoints, and generated large files from git |

## Prerequisites

- Linux with an NVIDIA GPU + drivers installed
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) (Anaconda or Miniconda)
- `git`, `ffmpeg`, `curl`
- [VIPE](https://github.com/nv-tlabs/vipe) — clone separately into `vipe-main/` (see Setup below). Not bundled in this repo.

`uv` is installed automatically by the script if it's missing — no manual step needed.
Gaussian Splatting is cloned automatically by the script if missing.

## Setup

**1. Clone this repo and VIPE into the same folder:**
```bash
git clone https://github.com/faafrii/VIPE-Gaussian-Splatting-Pipeline.git
cd VIPE-Gaussian-Splatting-Pipeline
git clone https://github.com/nv-tlabs/vipe vipe-main
```
(Gaussian Splatting is cloned automatically by the script if missing — no manual step needed for it.)

**2. Add your dataset.** Place your images in a folder named after your scene, inside VIPE's `images/` directory:
```bash
mkdir -p vipe-main/images/your_scene
cp /path/to/your/photos/*.jpg vipe-main/images/your_scene/
```

You should end up with:
```
VIPE-Gaussian-Splatting-Pipeline/
├── run_pipeline.sh
├── README.md
├── CHANGELOG.md
├── .gitignore
├── vipe-main/
│   └── images/
│       └── your_scene/        ← your dataset goes here
│           ├── frame_0001.jpg
│           ├── frame_0002.jpg
│           └── ...
└── gaussian-splatting/        ← auto-cloned on first run
```

**3. Run it** — see [Usage](#usage) below.

## Usage

### From a video
```bash
./run_pipeline.sh -i /path/to/video.mp4 -n scene_name
```

### From a directory of images
```bash
./run_pipeline.sh -I ./vipe-main/images/scene_name -n scene_name
```

### First run vs. later runs
The first run installs everything (conda envs, `uv sync`, builds CUDA
extensions, clones Gaussian Splatting) — expect it to take a while.
Later runs on a different dataset can skip setup:
```bash
./run_pipeline.sh -I ./vipe-main/images/scene_name -n scene_name --skip-setup
```

### Re-running a failed/partial run
Skip stages that already succeeded instead of starting over:
```bash
# VIPE data already generated — just (re)train and render
./run_pipeline.sh -I ./vipe-main/images/scene_name -n scene_name --skip-setup --skip-vipe

# Training already done — just re-render and rebuild the video
./run_pipeline.sh -I ./vipe-main/images/scene_name -n scene_name --skip-setup --skip-vipe --skip-train
```

### Forcing a clean environment rebuild
If a conda env may have been built before a fix (see `CHANGELOG.md`) was
in place, rebuild it from scratch:
```bash
./run_pipeline.sh -I ./vipe-main/images/scene_name -n scene_name --clean-env
```

## All flags

| Flag | Description | Default |
|---|---|---|
| `-i` | Input video file | — |
| `-I` | Input image directory | — |
| `-n` | Scene name (**required**) | — |
| `-v` | Path to VIPE repo | `./vipe-main` |
| `-g` | Path to Gaussian Splatting repo | `./gaussian-splatting` |
| `-o` | Output/workspace directory | `./pipeline_out` |
| `-t` | Training iterations | `30000` |
| `-f` | Render video framerate | `30` |
| `-d` | VIPE→COLMAP `depth_step` | `8` |
| `-m` | Max points fed to Gaussian Splatting (prevents GPU OOM) | `500000` |
| `--skip-setup` | Skip env creation / dependency install | off |
| `--skip-vipe` | Skip VIPE stage, reuse existing COLMAP data | off |
| `--skip-train` | Skip training, reuse existing checkpoint | off |
| `--clean-env` | Remove and recreate conda envs from scratch | off |
| `-h` | Show usage | — |

Use exactly one of `-i` or `-I`.

## Outputs

| Path | Contents |
|---|---|
| `vipe-main/vipe_test_out/` | VIPE outputs: depth, pose, mask, RGB, visualization video |
| `vipe-main/colmap_out/<scene>/` | COLMAP-format camera/sparse point data |
| `gaussian-splatting/output/<scene>/` | Trained Gaussian Splatting checkpoint + renders |
| `pipeline_out/<scene>_demo.mp4` | Final rendered demo video |

## Troubleshooting

Known issues and their root causes are documented in `CHANGELOG.md`
(conda/`uv` install, CUDA toolchain conflicts, path resolution bugs, etc).
If something fails, check there first before debugging from scratch.
