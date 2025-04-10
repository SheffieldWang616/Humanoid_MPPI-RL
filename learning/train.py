import torch
from model import MLPStatePredictor
from data_loader import StateActionDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os

def train_model(device='cuda'):
    state_csv = "data/2025-04-09_145305/states.csv"
    action_csv = "data/2025-04-09_145305/actions.csv"
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    dataset = StateActionDataset(state_csv, action_csv, return_delta=False)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


    model = MLPStatePredictor(state_dim=55, action_dim=21, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 200
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (input_features, target) in enumerate(train_loader):
            input_features = input_features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model.forward(input_features)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        if (epoch+1) % 10 == 0:
            # Save the model after each epoch
            model_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    print("Training complete.")
    # print first 5 predictions
    for i in range(3):
        input_features, target = dataset[i]
        input_features = torch.tensor(input_features).unsqueeze(0).to(device)
        output = model(input_features)
        print("Input: ", input_features)
        print("Target: ", target)
        print("Output: ", output)

if __name__ == "__main__":
    train_model()
