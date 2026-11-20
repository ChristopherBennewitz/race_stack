# PyTorch for the RL controller and On-Track-SysID. Runs at `docker compose build` (the Dockerfile calls it without
# arguments: CPU torch), or by hand inside a container / on a host.
# Usage: torch_setup.sh [none|cpu|gpu] [platform: x86|arm|jet]
set -eo pipefail

MODE="${1:-cpu}"
PLATFORM="${2:-x86}"
UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
    none)
        echo "Skipping the PyTorch install."
        exit 0
        ;;
    cpu)
        # On x86 take torch from the CPU index, so pip does not pull the CUDA wheels.
        # torch>=2.11 requires a setuptools newer than the 58.2.0 that ROS humble needs.
        if [[ "$(uname -m)" == "x86_64" ]]; then TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"; fi
        pip3 install "torch<2.11" $TORCH_INDEX
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
        echo "Error: unknown mode '$MODE' (expected none, cpu or gpu)." >&2
        exit 1
        ;;
esac

# Installing torch can bring in a newer setuptools, which breaks ROS 2 Humble's colcon --symlink-install
# (python_req.txt pins this version for that reason), so pin it again afterwards.
pip3 install "setuptools==58.2.0"
