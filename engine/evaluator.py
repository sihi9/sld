import torch
import torch.nn.functional as F
import os
from spikingjelly.activation_based import functional
from tqdm import tqdm
from torch.amp import autocast
from utils.monitoring import SpikeLogger


def run_final_evaluation_and_save(
    model: torch.nn.Module,
    val_loader,
    optimizer,
    scaler,
    device,
    amp,
    epochs,
    logger : SpikeLogger,
) -> None:
    """
    Evaluates the model and saves the final checkpoint with metrics and state.

    Args:
        model: Trained model.
        val_loader: Validation dataloader.
        optimizer: Optimizer instance.
        scaler: AMP GradScaler, or None.
        device: Torch device ('cuda', 'cpu', etc.).
        amp: Whether to use automatic mixed precision.
        epochs: Number of epochs trained.
        checkpoint_dir: Directory where checkpoint will be saved.
    """
    print("Running final evaluation on validation set...")
    final_loss, final_iou = evaluate(model, val_loader, device, use_amp=amp)
    print(f"Final Loss: {final_loss:.4f}, Final IoU: {final_iou:.4f}")
    logger.log_scalar("test/final_IoU", final_iou, step=epochs)
    logger.log_scalar("test/final_loss", final_loss, step=epochs)
    
    

    logger.save_checkpoint(name='checkpoint_final',
                           model=model,
                           optimizer=optimizer,
                           scaler=scaler,
                           epoch=epochs,
                           metrics={"final_iou": final_iou, "final_loss": final_loss},)
   
    best_checkpoint = os.path.join(logger.checkpoint_dir, "checkpoint_latest.pth") 
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    best_loss, best_iou = evaluate(model, val_loader, device, use_amp=amp)
    print(f"Best Loss: {best_loss:.4f}, Best IoU: {best_iou:.4f}")
    logger.log_scalar("test/best_IoU", best_iou, step=epochs)
    logger.log_scalar("test/best_loss", best_loss, step=epochs)


def evaluate(model, dataloader, device, loss_fn=None, use_amp=False):
    """
    Standard evaluation loop for validation or test.

    Args:
        model: Spiking segmentation model
        dataloader: PyTorch DataLoader yielding (input, label)
        device: 'cuda', 'cpu', or 'mps'
        loss_fn: Loss function (default: binary cross entropy)
        use_amp: Enable automatic mixed precision
    Returns:
        avg_loss: Mean loss over dataset
        avg_iou: Mean IoU over dataset
    """
    model.eval()
    model.to(device)

    if loss_fn is None:
        # Use raw logits; targets can be smoothed
        pos_weight = torch.tensor([5.0], device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss = 0.0
    total_iou = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            # [B, T, C, H, W] -> [T, B, C, H, W]
            inputs = inputs.permute(1, 0, 2, 3, 4).to(device)
            targets = targets.to(device)

            with autocast(device_type=device.split(':')[0]) if use_amp else torch.no_grad():
                outputs = model(inputs, return_logits=True) 
                loss = loss_fn(outputs, targets)

            iou = compute_batch_iou(outputs, targets, expect_logits=True)

            total_loss += loss.item()
            total_iou += iou
            total_batches += 1

            functional.reset_net(model)
             
            # Clear monitors
            if hasattr(model, 'output_monitor') and model.output_monitor is not None:
                model.output_monitor.clear_recorded_data()
            if hasattr(model, 'v_monitor') and model.v_monitor is not None:
                model.v_monitor.clear_recorded_data()
                

    avg_loss = total_loss / total_batches
    avg_iou = total_iou / total_batches

    return avg_loss, avg_iou


def compute_batch_iou(preds, targets, expect_logits=False, threshold=0.5, eps=1e-6):
    """
    Computes mean IoU for a batch of predictions and targets.

    Args:
        preds: Tensor of shape [B, 1, H, W]
        targets: Tensor of same shape
        expect_logits: If True, apply sigmoid to preds
        threshold: Threshold to binarize outputs
        eps: Small value to avoid division by zero
    """
    if expect_logits:
        preds = torch.sigmoid(preds)
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()

    intersection = (preds_bin * targets_bin).sum(dim=(1,2,3))
    union        = (preds_bin + targets_bin).clamp(0,1).sum(dim=(1,2,3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
