(参考) https://github.com/google-research/google-research/tree/master/bigger_better_faster
原ライセンス：Apache-2.0 license
詳細は[LICENSE](./LICENSE)を参照

## 手順
準備
```bash
set -eux

python3 -m venv .env
source .env/bin/activate

pip3 install -r requirements.txt
```

学習実行
```bash
python3 -m bbf.train \
    --agent=BBF \
    --gin_files=bbf/configs/BBF.gin \
    --base_dir=./bbf_result_$(date +%Y%m%d_%H%M%S) \
    --run_number=1
```
