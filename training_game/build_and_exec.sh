#!/bin/bash
set -eux

cd $(dirname $0)

# rm -rf build
# mkdir build && cd build
# cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
# cd ..

cmake --build build && ./Siv3DApp
