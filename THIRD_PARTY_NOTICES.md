# Third-party software and assets

The MIT license in `LICENSE` applies only to the original code and documentation in this repository. Third-party projects retain their own copyright and license terms.

## LeRobot

- Project: [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- Verified revision: `2774d9bd`
- License: Apache License 2.0
- Use in this project: dataset conversion, ACT training, preprocessing, postprocessing, and inference

LeRobot is installed as an external Python package and is not vendored in this repository.

## MuJoCo

- Project: [Google DeepMind MuJoCo](https://github.com/google-deepmind/mujoco)
- Verified Python package version: `3.8.1`
- License: Apache License 2.0
- Use in this project: simulation, rendering, contacts, and robot control

MuJoCo is installed as an external Python package and is not vendored in this repository.

## Unitree repositories and robot assets

- Project: [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
  - Verified revision: `7d6075f`
  - Provides the G1 robot descriptions and meshes used locally.
  - No top-level license file was present in the verified local checkout. Consult the upstream repository before redistributing any files or assets.
- Project: [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
  - Verified revision: `1eb6642`
  - License in the verified checkout: BSD 3-Clause License

Both repositories, including URDF/XML files and meshes, are excluded from this repository. Users must obtain them directly from Unitree and comply with their upstream terms.

## Other dependencies

PyTorch, NumPy, PyAV, OpenCV, Pillow, PyArrow, Safetensors, and their transitive dependencies retain their respective licenses. They are installed through the environment and are not redistributed here.

## Data and trained weights

Demonstration recordings, converted datasets, and trained model weights are excluded from Git. If distributed separately, their applicable data/model terms and SHA-256 checksums should accompany the download.
