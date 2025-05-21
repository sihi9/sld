import matplotlib.pyplot as plt
import torch
import numpy as np


def show_sample_triplet(input_seq, output_seq, label_seq, sample_idx=0, time_idx=0, figsize=(12, 4), cmap='gray'):
    """
    Plot one triplet of input / output / label frames.

    Args:
        input_seq: Tensor [T, B, 1, H, W]
        output_seq: Tensor [T, B, 1, H, W]
        label_seq: Tensor [T, B, 1, H, W]
        sample_idx: Which sample in the batch to visualize (0-based)
        time_idx: Which timestep in the sequence to visualize (0-based)
    """
    input_img = input_seq[time_idx, sample_idx, 0].cpu().numpy()
    output_img = output_seq[time_idx, sample_idx, 0].detach().cpu().numpy()
    label_img = label_seq[time_idx, sample_idx, 0].cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    axs[0].imshow(input_img, cmap=cmap)
    axs[0].set_title("Input")
    axs[0].axis("off")

    axs[1].imshow(output_img, cmap=cmap)
    axs[1].set_title("Predicted Output")
    axs[1].axis("off")

    axs[2].imshow(label_img, cmap=cmap)
    axs[2].set_title("Ground Truth")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
