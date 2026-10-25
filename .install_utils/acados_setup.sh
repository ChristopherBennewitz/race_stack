#! /bin/bash
# Builds acados v0.4.4 for the progress_mpc package (the controller manager's MPC mode).
# Runs at `docker compose build`, or by hand on a host.
# Usage: acados_setup.sh [install-path, default: /opt/acados]
set -e

ACADOS_ROOT="${1:-/opt/acados}"
if [ -f "$ACADOS_ROOT/lib/libacados.so" ]; then
    echo "acados already installed in $ACADOS_ROOT"
    exit 0
fi

git clone --branch v0.4.4 --depth 1 https://github.com/acados/acados.git "$ACADOS_ROOT"
cd "$ACADOS_ROOT"
git submodule update --init --recursive --depth 1
mkdir -p build && cd build
# Flags as in f1tenth_parallel_sim/install/install_acados.sh: modern CMake/GCC reject BLASFEO/HPIPM/qpOASES otherwise.
cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_FLAGS="-Wno-error=implicit-function-declaration -Wno-error=incompatible-pointer-types -Wno-error=implicit-int" \
  ..
make install -j"$(nproc)"
echo "acados installed to $ACADOS_ROOT"
