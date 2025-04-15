import torch
import numpy as np
from model import MLPStatePredictor, FeatureAttentionStatePredictor
from data_loader import StateActionDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import time

def train_model(device='cuda'):
    state_csv = "data/2025-04-09_145305/states.csv"
    action_csv = "data/2025-04-09_145305/actions.csv"
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    log_dir = "runs/train_logs"
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, time_str)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    dataset = StateActionDataset(state_csv, action_csv, return_delta=False)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    eval_dataset = StateActionDataset(state_csv, action_csv, return_delta=False, split='eval')
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Train dataset size: {len(dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    model = MLPStatePredictor(state_dim=55, action_dim=21, hidden_dim=256, use_batch_norm=True, dropout_rate=0.2, hidden_layers=3).to(device)
    # model = FeatureAttentionStatePredictor(state_dim=55, action_dim=21, hidden_dim=256, attn_layers=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    num_epochs = 5000

    model.train()
    loss_function = torch.nn.L1Loss()
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        if (epoch+1) % 100 == 0:
            # Save the model after each epoch
            model_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        # Evaluate the model on the evaluation dataset
        
        model.eval()
        with torch.no_grad():
            mean_diffs = []
            max_diffs = []
            losses = []
            for batch_idx, (input_features, target) in enumerate(eval_loader):
                input_features = input_features.to(device)
                target = target.to(device)
                output = model(input_features)
                loss = loss_function(output, target).cpu().numpy()
                losses.append(loss)

                output = output.cpu().numpy()
                diff = np.abs(output - target.cpu().numpy())
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)
                mean_diffs.append(mean_diff)
                max_diffs.append(max_diff)
                
            mean_diff = np.mean(mean_diffs)
            max_diff = np.mean(max_diffs)
            loss = np.mean(losses)
        print(f"Epoch [{epoch+1}/{num_epochs}], Eval Mean Diff: {mean_diff:.4f}, Max Diff: {max_diff:.4f} Loss: {loss.item():.4f}")
                
        writer.add_scalar("Eval/Mean_Diff", mean_diff, epoch)
        writer.add_scalar("Eval/Max_Diff", max_diff, epoch)
        writer.add_scalar("Eval/Loss", loss.item(), epoch)


    writer.close()
    print("Training complete.")
    # print first 5 predictions
    for i in range(3):
        model.eval()
        input_features, target = dataset[i]
        input_features = torch.tensor(input_features).unsqueeze(0).to(device)
        output = model(input_features).to('cpu')
        print("Input: ", input_features)
        print("Target: ", target)
        print("Output: ", output)

if __name__ == "__main__":
    train_model()
