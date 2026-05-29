#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Cloning ALE ==="
if [ ! -d "$SCRIPT_DIR/ALE" ]; then
    git clone --depth=1 https://github.com/Farama-Foundation/Arcade-Learning-Environment.git "$SCRIPT_DIR/ALE"
else
    echo "ALE already cloned, skipping."
fi

echo "=== Building ALE ==="
mkdir -p "$SCRIPT_DIR/ALE/build"
cmake -S "$SCRIPT_DIR/ALE" -B "$SCRIPT_DIR/ALE/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/deps" \
    -DALE_BUILD_CPP_LIB=ON \
    -DALE_USE_SDL=OFF \
    -DBUILD_SHARED_LIBS=OFF
cmake --build "$SCRIPT_DIR/ALE/build" --parallel "$(nproc)"
cmake --install "$SCRIPT_DIR/ALE/build"

echo "=== ALE installed to $SCRIPT_DIR/deps ==="

echo "=== Building pacman-neat ==="
# rm -rf "$SCRIPT_DIR/build"
mkdir -p "$SCRIPT_DIR/build"
cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$SCRIPT_DIR/deps"
cmake --build "$SCRIPT_DIR/build" --parallel "$(nproc)"

echo ""
echo "Done! Run:"
ROM=$(python3 -c "import ale_py; from pathlib import Path; print(Path(ale_py.__file__).parent / 'roms/pacman.bin')" 2>/dev/null || echo "/path/to/pacman.bin")
echo "  ./build/pacman_neat $ROM"
