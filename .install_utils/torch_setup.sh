#! /bin/bash
# Optional PyTorch install for the RL controller. Runs at `docker compose build`
# (make build RL=cpu|gpu), or by hand inside a container / on a host.
# Usage: torch_setup.sh <none|cpu|gpu> [platform: x86|arm|jet]
set -e

MODE="${1:-none}"
PLATFORM="${2:-x86}"
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
    none)
        echo "RL disabled: skipping PyTorch install."
        exit 0
        ;;
    cpu)
        pip3 install -r "$UTILS_DIR/torch_cpu_req.txt"
        ;;
    gpu)
        if [[ "$PLATFORM" == "jet" ]]; then
            # Jetson needs NVIDIA's JetPack-matched wheels; set the index for your JetPack version.
            if [[ -n "$TORCH_GPU_INDEX_URL" ]]; then
                pip3 install torch --index-url "$TORCH_GPU_INDEX_URL"
            elif python3 -c "import torch" 2>/dev/null; then
                echo "Using the torch already present in the base image."
            else
                echo "Error: no torch in the base image. Set TORCH_GPU_INDEX_URL to a JetPack-matching wheel index." >&2
                exit 1
            fi
        else
            pip3 install -r "$UTILS_DIR/torch_gpu_req.txt"
        fi
        ;;
    *)
        echo "Error: unknown RL mode '$MODE' (expected none, cpu or gpu)." >&2
        exit 1
        ;;
esac

# Installing torch can bring in a newer setuptools, which breaks ROS 2 Humble's colcon --symlink-install
# (python_req.txt pins this version for that reason), so pin it again afterwards.
pip3 install "setuptools==58.2.0"
