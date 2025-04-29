import torch
import numpy as np
from model import MLPStatePredictor, FeatureAttentionStatePredictor
from data_loader import StateActionDataset, MultiTrajectoryDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import time

def train_model(device='cuda'):
    state_csv = "data_cartpole/states"
    action_csv = "data_cartpole/actions"
    ckpt_dir = "checkpoints_cartpole"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize TensorBoard writer
    log_dir = "runs/cartpole_train_logs"
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, time_str)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    dataset_type = "delta"
    dataset_normalize = False
    dataset_random_split = True
    dataset_smooth_window_size = 0
    train_ratio = 0.9

    dataset = MultiTrajectoryDataset(state_csv, action_csv, return_type=dataset_type, normalize=dataset_normalize, train_ratio=train_ratio,
                                        random_split=dataset_random_split, smooth_window_size=dataset_smooth_window_size)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    eval_dataset = MultiTrajectoryDataset(state_csv, action_csv, return_type=dataset_type, normalize=dataset_normalize, train_ratio=train_ratio, split='eval',
                                        random_split=dataset_random_split, smooth_window_size=dataset_smooth_window_size,)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Train dataset size: {len(dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # model = MLPStatePredictor(state_dim=55, action_dim=21, hidden_dim=512, use_batch_norm=True, dropout_rate=0.2, hidden_layers=6).to(device)
    model = FeatureAttentionStatePredictor(
            state_dim=4, action_dim=1, hidden_dim=64, num_heads=4, attn_layers=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)
    # model.load_state_dict(torch.load("checkpoints_cartpole/model_best.pth"))
    eval_loss_min = float('inf')
    model.train()
    loss_function = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (input_features, target) in enumerate(train_loader):
            input_features = input_features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model.forward(input_features)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            # Log the loss to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), global_step)

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        if (epoch+1) % 10 == 0:
            # Save the model after each epoch
            model_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        # Evaluate the model on the evaluation dataset
        scheduler.step()
        writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        model.eval()
        with torch.no_grad():
            mean_diffs = []
            max_diffs = []
            mean_pct_diffs = []
            max_pct_diffs = []
            losses = []
            diffs = []
            for batch_idx, (input_features, target) in enumerate(eval_loader):
                input_features = input_features.to(device)
                target = target.to(device)
                output = model(input_features)
                loss = loss_function(output, target).cpu().numpy()
                losses.append(loss)

                target = target.cpu().numpy()
                output = output.cpu().numpy()
                input_features = input_features.cpu().numpy()
                diff = np.abs(output - target)
                assert diff.shape[1] == 4, "Output shape mismatch, expected 4 columns but target has shape {} and output has shape {}".format(target.shape, output.shape)
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)

                mean_pct_diff = np.mean(np.abs(diff / input_features[:, :diff.shape[1]]))
                max_pct_diff = np.max(np.abs(diff / input_features[:, :diff.shape[1]]))

                diffs.append(diff)
                mean_diffs.append(mean_diff)
                max_diffs.append(max_diff)
                mean_pct_diffs.append(mean_pct_diff)
                max_pct_diffs.append(max_pct_diff)

            mean_diff = np.mean(mean_diffs)
            max_diff = np.mean(max_diffs)
            loss = np.mean(losses)
            mean_pct_diff = np.mean(mean_pct_diffs)
            max_pct_diff = np.mean(max_pct_diffs)
            if loss < eval_loss_min:
                eval_loss_min = loss
                # Save the model if the loss is lower than the previous minimum
                model_path = os.path.join(ckpt_dir, "model_best.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved to {model_path}")
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Eval Mean Diff: {mean_diff:.4f}, Max Diff: {max_diff:.4f} Loss: {loss.item():.4f}")

        writer.add_scalar("Eval/Mean_Diff", mean_diff, epoch)
        writer.add_scalar("Eval/Max_Diff", max_diff, epoch)
        writer.add_scalar("Eval/Mean_Pct_Diff", mean_pct_diff, epoch)
        writer.add_scalar("Eval/Max_Pct_Diff", max_pct_diff, epoch)
        writer.add_scalar("Eval/Loss", loss.item(), epoch)
        
        diffs = np.vstack(diffs)
        # log each diff to tensorboard, each diff has its own column
        for i in range(diffs.shape[1]):
            writer.add_scalar(f"Diffs/Column_{i}", np.mean(diffs[:, i]), epoch)
            writer.add_scalar(f"Diffs/Max_Column_{i}", np.max(diffs[:, i]), epoch)

    writer.close()
    print("Training complete.")
    # Save the final model
    model_path = os.path.join(ckpt_dir, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    # print first 5 predictions
    for i in range(3):
        model.eval()
        input_features, target = dataset[i]
        input_features = torch.tensor(input_features).unsqueeze(0).to(device)
        output = model(input_features).to('cpu')
        pct_diff = (output - target) / target
        print(f"Input: {input_features}, Target: {target}, Output: {output}, Pct Diff: {pct_diff}")

if __name__ == "__main__":
    train_model()
