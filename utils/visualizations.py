import io
import os
import cv2
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import torch
import numpy as np
from spikingjelly.activation_based import functional, layer, neuron
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

from utils.monitoring import SpikeLogger

def visualize_random_batch(model, dataloader, device, sample_idx=0, n=3, time_idx=0, logger=None, step=0):
    """
    Visualizes predictions and optionally logs to TensorBoard.

    Args:
        logger: SpikeLogger or None. If provided, logs the figure instead of showing.
        step: Global step or epoch for TensorBoard.
    """
    model.eval()
    with torch.no_grad():
        for input_seq, label_seq in dataloader:
            input_seq = input_seq.permute(1, 0, 2, 3, 4).to(device)  # [T, B, C, H, W]
            label_seq = label_seq.to(device)

            output_seq = model(input_seq)  # [B, 1, H, W]

            fig = show_sample_triplet(input_seq.cpu(), 
                                      output_seq.cpu(),
                                      label_seq.cpu(),
                                      n=n)
            if logger:
                logger.writer.add_figure("predictions/sample_triplet", fig, global_step=step)
            else:
                plt.show()
            break  

    functional.reset_net(model)


def visualize_batch_predictions(
    input_seq: torch.Tensor,
    label_seq: torch.Tensor,
    output_seq: torch.Tensor,
    logger: Optional[SpikeLogger] = None,
    step: int = 0,
    title_tag: str = "batch_sample"
):
    """
    Visualizes model predictions for a specific batch.

    Args:
        input_seq: [T, B, C, H, W]
        label_seq: [B, 1, H, W]
        output_seq: [B, 1, H, W]
    """
    
    fig = show_sample_triplet(input_seq.cpu(),
                            output_seq.cpu(),
                            label_seq.cpu(),
                            n=label_seq.shape[0])

    if logger:
        logger.writer.add_figure(f"predictions/{title_tag}", fig, global_step=step)
    else:
        plt.show()
    

def show_sample_triplet(input_seq, output_seq, label_seq, n=3, figsize=(8, 8), cmap='gray'):
    """
    Plot one triplet of input / output / label frames.

    Returns:
        fig: Matplotlib Figure object
    """
    fig, axs = plt.subplots(n, 3, figsize=figsize)
    axs = axs if n > 1 else [axs]
    
    for i in range(n):
        input_img = input_seq[-1, i, 0].cpu().numpy()
        output_img = output_seq[i, 0].detach().cpu().numpy()
        label_img = label_seq[i, 0].cpu().numpy()

        axs[i][0].imshow(input_img, cmap=cmap)
        axs[i][0].set_title(f"Input #{i}")
        axs[i][0].axis("off")

        axs[i][1].imshow(output_img, cmap=cmap)
        axs[i][1].set_title("Predicted")
        axs[i][1].axis("off")

        axs[i][2].imshow(label_img, cmap=cmap)
        axs[i][2].set_title("Ground Truth")
        axs[i][2].axis("off")
    plt.tight_layout()
    return fig


def visualize_predictions_video(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = "./outputs/test_video",
    all_timesteps: bool = True,
    fps: int = 10,
    threshold: float = 0.5,
):
    """
    Generates a video from the entire test set sequence with overlaid predictions and ground truth.
    - If all_timesteps=True, visualize all frames per sample.
    - Otherwise, visualize only the final frame of each sample.

    Args:
        model: Trained PyTorch model
        dataloader: Test DataLoader
        device: Torch device
        save_dir: Output directory
        all_timesteps: Whether to use all T timesteps or just the last
        fps: Video FPS
        threshold: Output threshold for binary prediction
    """
    print("🔍 Generating test set video...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    frame_paths = []
    with torch.no_grad():
        idx = 0
        for batch in dataloader:
            inputs, labels = batch  # inputs: [B, T, 1, H, W], labels: [B, 1, H, W]
            B, T, _, H, W = inputs.shape

            inputs = inputs.to(device)
            labels = labels.to(device)
            input_seq = inputs.permute(1, 0, 2, 3, 4)  # [T, B, 1, H, W]
            outputs = model(input_seq)                # [B, 1, H, W]
            preds = (outputs > threshold).float()

            for b in range(B):
                x = inputs[b]       # (T, 1, H, W)
                y = labels[b]       # (1, H, W)
                pred = preds[b]     # (1, H, W)

                for t in range(T if all_timesteps else T-1, T):
                    img = x[t, 0].cpu().numpy()
                    img_rgb = np.stack([img] * 3, axis=-1) * 255
                    img_rgb = img_rgb.astype(np.uint8)
                    
                    if t == T - 1:
                        # Get binary masks
                        y_np = y[0].cpu().numpy().astype(bool)
                        pred_np = pred[0].cpu().numpy().astype(bool)

                        # Convert grayscale image to BGR and normalize to [0, 1]
                        img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)
                        img_rgb /= 255.0 if img_rgb.max() > 1.0 else 1.0

                        # Define colors in BGR (OpenCV format)
                        green  = np.array([0.0, 1.0, 0.0])  # GT only
                        red    = np.array([0.0, 0.0, 1.0])  # Prediction only
                        yellow = np.array([0.0, 1.0, 1.0])  # Overlap (GT & prediction)

                        alpha = 0.4

                        # Create overlay map
                        overlay = np.zeros_like(img_rgb)

                        gt_only = y_np & ~pred_np
                        pred_only = pred_np & ~y_np
                        overlap = y_np & pred_np

                        overlay[gt_only] = green
                        overlay[pred_only] = red
                        overlay[overlap] = yellow

                        mask = gt_only | pred_only | overlap
                        mask3 = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

                        # Alpha blend
                        img_rgb[mask3] = (1 - alpha) * img_rgb[mask3] + alpha * overlay[mask3]

                        # Convert back to uint8
                        img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8)

                    img_path = os.path.join(save_dir, f"frame_{idx:05d}.png")
                    cv2.imwrite(img_path, img_rgb)
                    frame_paths.append(img_path)
                    idx += 1

    # Combine into video
    fourcc = cv2.VideoWriter_fourcc(*'I420')  # Or 'FFV1' for truly lossless
    video_path = os.path.join(save_dir, "testset_video.avi")
    sample_img = cv2.imread(frame_paths[0])
    height, width, _ = sample_img.shape
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for path in frame_paths:
        frame = cv2.imread(path)
        out.write(frame)
    out.release()
    print(f"✅ Saved test set video with {len(frame_paths)} frames to: {video_path}")

    

def visualize_weights(
    model: nn.Module, 
    logger: SpikeLogger, 
    step: int,
    layer_logging_prefs: Optional[Dict[str, Literal["histogram", "heatmap"]]] = None
) -> None:
    """
    Logs weights of the model based on type and user-defined preferences.

    Args:
        model: The PyTorch model.
        logger: SpikeLogger with SummaryWriter and vis_interval.
        step: Current epoch or global step.
        layer_logging_prefs: Optional dict {layer_name_substring: "histogram" | "heatmap"}
    """
    if logger.vis_interval is None or logger.vis_interval <= 0:
        return
    if (step + 1) % logger.vis_interval != 0:
        return

    print("🔍 Visualizing weights...")

    log_conv_kernels(model, logger, step)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, layer.Linear)):
            mode = "histogram"  # default

            # check user-defined overrides
            if layer_logging_prefs:
                for key, pref in layer_logging_prefs.items():
                    if key in name:
                        mode = pref
                        break

            if mode == "histogram":
                log_linear_weights_histogram_named(name, module, logger, step)
            elif mode == "heatmap":
                log_linear_weights_heatmap_named(name, module, logger, step)
                
    log_tau_per_plif_layer(model, logger, step)
    


def log_conv_kernels(model: nn.Module, logger: SpikeLogger, step: int, max_kernels: int = 32) -> None:
    """
    Logs selected 2D kernels from convolutional layers.
    
    Args:
        model: The PyTorch model.
        logger: TensorBoard logger.
        step: Current training step or epoch.
        max_kernels: Max number of 2D kernels to display per layer.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, layer.Conv2d)):
            weight = module.weight.data.clone().cpu()  # [out_ch, in_ch, H, W]
            out_ch, in_ch, h, w = weight.shape

            # Flatten all individual 2D kernels: shape [out_ch * in_ch, H, W]
            kernels = weight.view(-1, h, w)

            # Normalize each kernel to [0,1]
            min_vals = torch.amin(kernels, dim=(1, 2), keepdim=True)
            max_vals = torch.amax(kernels, dim=(1, 2), keepdim=True)
            kernels = (kernels - min_vals) / (max_vals - min_vals + 1e-5)

            # Limit to max_kernels
            if kernels.shape[0] > max_kernels:
                idx = torch.linspace(0, kernels.shape[0] - 1, steps=max_kernels).long()
                kernels = kernels[idx]

            # Add channel dim: [N, 1, H, W]
            kernels = kernels.unsqueeze(1)

            grid = make_grid(kernels, nrow=int(max_kernels**0.5), normalize=False, pad_value=1)
            logger.writer.add_image(f"Weights/Kernels/{name}", grid, global_step=step)

def log_linear_weights_histogram_named(name: str, module: nn.Module, logger: SpikeLogger, step: int) -> None:
    logger.writer.add_histogram(f"Weights/Histogram/{name}", module.weight.data, global_step=step)
    if module.bias is not None:
        logger.writer.add_histogram(f"Weights/Bias/{name}", module.bias.data, global_step=step)


def log_linear_weights_heatmap_named(name: str, module: nn.Module, logger: SpikeLogger, step: int) -> None:
    weight = module.weight.data.cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(weight, aspect='auto', cmap='viridis')
    ax.set_title(f"Heatmap - {name}")
    ax.set_xlabel("Input Neurons")
    ax.set_ylabel("Output Neurons")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    import PIL.Image
    image = PIL.Image.open(buf)
    logger.writer.add_image(f"Weights/Heatmap/{name}", np.array(image), step, dataformats='HWC')
    plt.close(fig)

def log_tau_per_plif_layer(model: nn.Module, logger: SpikeLogger, step: int):
    """
    Logs a separate tau histogram for each PLIFNode layer.
    """
    print("📈 Logging PLIFNode taus...")

    for name, module in model.named_modules():
        if isinstance(module, neuron.ParametricLIFNode):
            tau = 1.0 / module.w.sigmoid().detach()
            logger.writer.add_histogram(f"NeuronTau/{name}", tau, global_step=step)
