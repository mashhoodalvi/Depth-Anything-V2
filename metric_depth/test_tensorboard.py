import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter


def main():
    # Azure ML automatically collects ./outputs
    log_dir = "./metric_depth/outputs"
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    epochs = 5
    iters_per_epoch = 20

    global_step = 0

    for epoch in range(epochs):
        # ---- training loop ----
        for i in range(iters_per_epoch):
            # dummy loss that decreases
            loss = torch.exp(torch.tensor(-0.1 * global_step))

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", 1e-4 * (1 - global_step / (epochs * iters_per_epoch)), global_step)

            print(f"Epoch {epoch} | Iter {i} | Loss {loss.item():.4f}")

            global_step += 1
            time.sleep(0.05)

        # ---- validation ----
        val_abs_rel = 0.5 / (epoch + 1)
        val_rmse = 5.0 / (epoch + 1)

        writer.add_scalar("val/abs_rel", val_abs_rel, epoch)
        writer.add_scalar("val/rmse", val_rmse, epoch)

    writer.close()


if __name__ == "__main__":
    main()