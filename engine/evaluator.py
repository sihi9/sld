import torch
import torch.nn.functional as F
import os
from spikingjelly.activation_based import functional
from tqdm import tqdm
from torch.amp import autocast


def run_final_evaluation_and_save(
    model: torch.nn.Module,
    val_loader,
    optimizer,
    scaler,
    cfg,
    checkpoint_dir: str
) -> None:
    """
    Evaluates the model and saves the final checkpoint with metrics and state.

    Args:
        model: Trained model.
        val_loader: Validation dataloader.
        optimizer: Optimizer instance.
        scaler: AMP GradScaler, or None.
        cfg: Configuration object.
        checkpoint_dir: Directory where checkpoint will be saved.
    """
    print("Running final evaluation on validation set...")
    final_loss, final_iou = evaluate(model, val_loader, cfg.train.device, use_amp=cfg.train.amp)
    print(f"Final Loss: {final_loss:.4f}, Final IoU: {final_iou:.4f}")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'epoch': cfg.train.epochs,
        'final_loss': final_loss,
        'final_iou': final_iou
    }

    path = os.path.join(checkpoint_dir, 'checkpoint_final.pth')
    torch.save(checkpoint, path)
    print(f"Final checkpoint saved to {path}")


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
        loss_fn = F.binary_cross_entropy

    total_loss = 0.0
    total_iou = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            # [B, T, C, H, W] -> [T, B, C, H, W]
            inputs = inputs.permute(1, 0, 2, 3, 4).to(device)
            targets = targets.to(device)

            with autocast(device_type=device.split(':')[0]) if use_amp else torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            iou = compute_batch_iou(outputs, targets)

            total_loss += loss.item()
            total_iou += iou
            total_batches += 1

            functional.reset_net(model)

    avg_loss = total_loss / total_batches
    avg_iou = total_iou / total_batches

    return avg_loss, avg_iou


def compute_batch_iou(preds, targets, threshold=0.5, eps=1e-6):
    """
    Computes mean IoU for a batch of predictions and targets.

    Args:
        preds: Tensor of shape [B, 1, H, W]
        targets: Tensor of same shape
        threshold: Threshold to binarize outputs
    """
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()

    intersection = (preds_bin * targets_bin).sum(dim=(1,2,3))
    union        = (preds_bin + targets_bin).clamp(0,1).sum(dim=(1,2,3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
