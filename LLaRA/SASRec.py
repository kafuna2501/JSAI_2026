import time
import threading

def keep_alive():
    while True:
        print("Keeping session alive...")
        time.sleep(600)  # 10分間隔で出力

# バックグラウンドスレッドを起動
thread = threading.Thread(target=keep_alive, daemon=True)  # daemon=True でバックグラウンド実行
thread.start()

print("バックグラウンドでセッションを維持中...")

!PYTHONPATH=/content python -u /content/LLaRA/main.py \
--mode test \
--llm_path "/content/drive/MyDrive/Uozumi/llama-2-7b" \
--rec_model_path ./rec_model/retail.pt \
--ckpt_path "/content/drive/MyDrive/Uozumi/JSAI_2026/checkpoints/sasrec_541/last.ckpt" \
--output_dir "/content/drive/MyDrive/Uozumi/JSAI_2026/output/sasrec_541/" \
--data_dir "/content/drive/MyDrive/Uozumi/JSAI_2026/data/ref/retail/" \
--lr_warmup_start_lr 7e-6 \
--lr 7e-4 \
--lr_decay_min_lr 7e-6 \
--max_epochs 5 \
--rec_size 64
