#!/usr/bin/env bash
# Build mdrender -- MilkDrop presets rendered natively by projectM 4 (LGPL-2.1, built here as a shared library) on an
# off-screen EGL context -- into ~/.cache/focalcut/bin. Needs git, cmake, g++ and the EGL / GL headers
# (Debian/Ubuntu: libegl-dev libgl-dev). focalcut's overlay runs this the first time it needs the MilkDrop layer.
set -euo pipefail
PM_REV=eae22de40144ee8ded92f3c04b93a4c2f82adcc0   # projectM master: projectm_set_frame_time (offline rendering)
root=${FOCALCUT_CACHE:-$HOME/.cache/focalcut}
src=$root/projectm pm=$root/projectm-install bin=$root/bin
here=$(cd "$(dirname "$0")" && pwd)
if [ ! -f "$pm/.rev" ] || [ "$(cat "$pm/.rev")" != "$PM_REV" ]; then
  rm -rf "$src" "$pm"
  git clone -q https://github.com/projectM-visualizer/projectm.git "$src"
  git -C "$src" checkout -q "$PM_REV"
  git -C "$src" submodule update -q --init --depth 1
  cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DENABLE_PLAYLIST=OFF \
    -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$pm" > /dev/null
  cmake --build "$src/build" -j"$(nproc)" > /dev/null
  cmake --install "$src/build" > /dev/null
  echo "$PM_REV" > "$pm/.rev"
fi
mkdir -p "$bin"
lib=$(dirname "$(find "$pm" -name 'libprojectM-4.so' | head -1)")
g++ -O2 -std=c++17 "$here/mdrender.cpp" -o "$bin/mdrender.part" -I"$pm/include" -L"$lib" -Wl,-rpath,"$lib" \
  -lprojectM-4 -lEGL -lGL
mv "$bin/mdrender.part" "$bin/mdrender"
echo "$bin/mdrender"
