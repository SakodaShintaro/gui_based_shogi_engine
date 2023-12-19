#!/bin/bash
set -eux

cd $(dirname $0)/../build-Release
cmake ../ -DCMAKE_BUILD_TYPE=Release
cmake --build ./ --config Release --target all --
./solve_grid_world_by_table
python3 ../python/plot_grid_world_log.py ./grid_world_log.tsv
