#!/usr/bin/env bash
set -euo pipefail

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j"$(nproc)"
cd ..
cp build/heat_core*.so .
echo "✓ heat_core built → $(ls heat_core*.so)"
