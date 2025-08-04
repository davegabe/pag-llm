import os
import torch
import shutil
import csv

base_ckpt_dir = "checkpoints/tinystories-pretokenized-base"
output_dir = "best-checkpoints"
log_file = os.path.join(output_dir, "best_checkpoints.csv")

os.makedirs(output_dir, exist_ok=True)

# Prepare CSV log
with open(log_file, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["method", "best_checkpoint", "val_loss"])

    for method in sorted(os.listdir(base_ckpt_dir)):
        if method == "ngram_cache":
            continue

        ckpt_dir = os.path.join(base_ckpt_dir, method)
        if not os.path.isdir(ckpt_dir):
            continue

        best_score = float("inf")
        best_path = None

        for file in sorted(os.listdir(ckpt_dir)):
            ckpt_path = os.path.join(ckpt_dir, file)
            if os.path.isdir(ckpt_path) or not (file.startswith("model-") and file.endswith(".ckpt")):
                continue

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                for cb_data in ckpt.get("callbacks", {}).values():
                    score = cb_data.get("best_model_score")
                    if score is not None:
                        score = score.item()
                        if score < best_score:
                            best_score = score
                            best_path = ckpt_path
            except Exception as e:
                print(f"Failed to load {file} in {method}: {e}")

        if best_path:
            out_ckpt_name = f"best-{method}.ckpt"
            out_ckpt_path = os.path.join(output_dir, out_ckpt_name)

            if not os.path.exists(out_ckpt_path):
                shutil.copy(best_path, out_ckpt_path)
                print(f"Copied best checkpoint for '{method}' to: {out_ckpt_path} (val/loss = {best_score:.4f})")
            else:
                print(f"Skipped copy for '{method}' — already exists.")

            writer.writerow([method, os.path.basename(best_path), f"{best_score:.4f}"])
        else:
            print(f"No valid checkpoint found for '{method}'")
