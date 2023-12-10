#!/bin/bash
set -eux

cd $(dirname $0)/../build-Release

rm -rf ./data

# 0から10までループ
for i in `seq 0 0`
do
    ./main_for_play
    # 存在すれば移動
    if [ -e ./data/offline_training ]; then
        mv ./data/offline_training ./data/offline_training_$(expr $i - 1)
    fi
    ./main_for_offline_training
    mv ./data/play ./data/play_${i}
done
