#!/bin/bash
set -eux

cd $(dirname $0)/../build-Release
cmake ../ -DCMAKE_BUILD_TYPE=Release
cmake --build ./ --config Release --target all --
./play_decision_transformer
