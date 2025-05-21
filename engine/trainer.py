import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from tqdm import tqdm
from engine.evaluator import evaluate
from utils.monitoring import SpikeLogger

def train(model, train_loader, val_loader, optimizer, device,
          loss_fn=None, scaler=None, epochs=10, use_amp=False, logger=None):
    """
    Trains the model and evaluates on validation set each epoch.
    """
    from torch.amp import autocast
    model.to(device)
    model.train()

    if loss_fn is None:
        loss_fn = F.binary_cross_entropy

    best_val_iou = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        for inputs, targets in loop:
            inputs = inputs.permute(1, 0, 2, 3, 4).to(device)   # [T, B, C, H, W]
            targets = targets.permute(1, 0, 2, 3, 4).to(device)

            optimizer.zero_grad()

            # todo: use amp
            outputs = model(inputs)
            outputs = outputs.mean(dim=0)  # Average over time steps
            
            loss = loss_fn(outputs, targets[-1])  # Use the last time step for loss calculation
            loss.backward()
            optimizer.step()
            

            functional.reset_net(model)
            running_loss += loss.item()
            total_batches += 1
            loop.set_postfix(train_loss=running_loss / total_batches)


        if logger is not None:
            from utils.monitoring import log_from_monitors
            log_from_monitors(model, logger, epoch)

        print("starting validation...")
        # Validation after each epoch
        val_loss, val_iou = evaluate(model, val_loader, device, loss_fn, use_amp)
        print(f"Validation - Loss: {val_loss:.4f}, Mean IoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print("🟢 New best model found!")

        # Switch back to train mode for next epoch
        model.train()