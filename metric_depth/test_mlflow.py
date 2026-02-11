import time
import math
import mlflow


def main():
    # Azure ML automatically initializes MLflow
    #mlflow.set_experiment("mlflow-dummy-test")

    epochs = 5
    iters_per_epoch = 20
    global_step = 0

    # log hyperparameters
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("iters_per_epoch", iters_per_epoch)
    mlflow.log_param("lr", 1e-4)

    for epoch in range(epochs):
        # ---- training loop ----
        for i in range(iters_per_epoch):
            loss = math.exp(-0.1 * global_step)
            lr = 1e-4 * (1 - global_step / (epochs * iters_per_epoch))

            mlflow.log_metric("train_loss", loss, step=global_step)
            mlflow.log_metric("lr", lr, step=global_step)

            print(f"Epoch {epoch} | Iter {i} | Loss {loss:.4f}")

            global_step += 1
            time.sleep(0.05)

        # ---- validation ----
        val_abs_rel = 0.5 / (epoch + 1)
        val_rmse = 5.0 / (epoch + 1)

        mlflow.log_metric("val_abs_rel", val_abs_rel, step=epoch)
        mlflow.log_metric("val_rmse", val_rmse, step=epoch)

    print("Training finished")


if __name__ == "__main__":
    main()