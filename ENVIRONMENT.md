# Reproducible environment

## Verified versions

| Component | Local simulation/inference | AutoDL training |
|---|---|---|
| Python | 3.12.14 | 3.12 |
| MuJoCo | 3.8.1 | not required for offline training |
| LeRobot | 0.6.2 dev, commit `2774d9bd` | editable installation |
| PyTorch | 2.11.0+cpu | 2.8.0+cu128 |
| GPU | CPU inference | NVIDIA RTX 4090D, 24 GB |

PyTorch is intentionally absent from `requirements.txt`: its installation command depends on whether the machine uses CPU, CUDA 12.8, or another CUDA version. Install the appropriate official PyTorch build first.

## Local setup

Create an environment and install the direct dependencies:

```bash
conda create -n g1-cup-act python=3.12 -y
conda activate g1-cup-act

# Install a platform-appropriate PyTorch build first.
pip install -r requirements.txt
```

Install the exact LeRobot source revision used by this project:

```bash
git clone https://github.com/huggingface/lerobot.git ../lerobot
git -C ../lerobot checkout 2774d9bd
pip install -e ../lerobot
```

The robot assets are upstream repositories and are excluded from this portfolio repository. Restore the verified revisions with:

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git unitree_ros
git -C unitree_ros checkout 7d6075f

git clone https://github.com/unitreerobotics/unitree_mujoco.git unitree_mujoco
git -C unitree_mujoco checkout 1eb6642
```

Run a basic scene and camera check:

```bash
MUJOCO_GL=egl python g1_cup_minimal.py --headless
```

Run the local ACT loading smoke test after placing the checkpoint at the path described in the README:

```bash
python act_inference_smoke_test.py
```

## Training environment notes

The recorded dataset was trained on AutoDL with an RTX 4090D. The relevant cache paths were:

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_LEROBOT_HOME=/root/autodl-tmp/huggingface/lerobot
```

These paths are examples for the original training container and are not required on another machine.

## Large artifacts

The following artifacts are intentionally excluded from Git:

- raw NPZ demonstrations and RGB videos;
- converted LeRobot dataset;
- ACT model weights and checkpoints;
- raw rollout diagnostics and logs;
- intermediate debug images and videos.

Model and dataset archives should be distributed separately with their `.sha256` files. Do not commit local Hugging Face tokens, SSH keys, environment-variable dumps, or AutoDL credentials.
