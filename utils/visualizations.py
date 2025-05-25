import matplotlib.pyplot as plt
import torch
import numpy as np
from spikingjelly.activation_based import functional



def visualize_predictions(model, dataloader, device, sample_idx=0, n = 3, time_idx=0):
    """
    Runs a single batch through the model and visualizes predictions.
    """

    model.eval()
    with torch.no_grad():
        for input_seq, label_seq in dataloader:
            input_seq = input_seq.permute(1, 0, 2, 3, 4).to(device)  # [T, B, C, H, W]
            label_seq = label_seq.permute(1, 0, 2, 3, 4).to(device)

            output_seq = model(input_seq)  # [T, B, 1, H, W]
            #output_prob = torch.sigmoid(output_seq)

            show_sample_triplet(input_seq.cpu(), 
                                output_seq.cpu(),
                                label_seq.cpu(),
                                n=n)
            break  

    functional.reset_net(model)


def show_sample_triplet(input_seq, output_seq, label_seq, n=3, figsize=(8, 8), cmap='gray'):
    """
    Plot one triplet of input / output / label frames.

    Args:
        input_seq: Tensor [T, B, 1, H, W]
        output_seq: Tensor [T, B, 1, H, W]
        label_seq: Tensor [T, B, 1, H, W]
        n: number of samples to visualize
    """
    fig, axs = plt.subplots(n, 3, figsize=figsize)
    axs = axs if n > 1 else [axs]
    
    for i in range(n):
        input_img = input_seq[0, i, 0].cpu().numpy()
        output_img = output_seq[i, 0].detach().cpu().numpy()
        label_img = label_seq[0, i, 0].cpu().numpy()

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
    plt.show()
