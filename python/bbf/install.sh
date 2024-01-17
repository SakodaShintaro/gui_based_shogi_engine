#!/bin/bash
set -eux

python3 -m venv .env
source .env/bin/activate

pip3 install -r requirements.txt

export PYTHONPATH="$(readlink -f ../):$PYTHONPATH"

python3 -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF.gin \
    --base_dir=/tmp/online_rl/bbf \
    --run_number=1
