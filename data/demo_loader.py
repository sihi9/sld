import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class DemoSegmentationDataset(Dataset):
    def __init__(self, 
                 num_samples=100, 
                 time_steps=10, 
                 input_size=(128, 128), 
                 line_width=3, 
                 moving=False,
                 noise=0.1,
                 heavy_noise=0.4,
                 heavy_noise_prob=0.3):
        """
        Initializes the dataset.
        Args:
            num_samples: Number of samples in the dataset.
            time_steps: Number of time steps in each sample.
            input_size: Tuple (H, W) for the height and width of the images.
            line_width: Width of the vertical line to draw.
            moving: If True, produces moving lines; otherwise static lines.
        """
        super().__init__()
        self.num_samples = num_samples
        self.T = time_steps
        self.H, self.W = input_size
        self.line_width = line_width
        self.moving = moving  # If True, produces moving lines; otherwise static lines
        self.noise = noise
        self.heavy_noise = heavy_noise
        self.heavy_noise_prob = heavy_noise_prob

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.moving:
            input_tensor, label_tensor = self.produce_moving_line(heavy_noise_prob=self.heavy_noise_prob,
                                                                  heavy_noise=self.heavy_noise)
        else:
            input_tensor, label_tensor = self.produce_static_line(noise=self.noise)
        return input_tensor, label_tensor            
    
    
    def produce_static_line(self, noise=0.1):
        input_tensor = np.zeros((self.T, 1, self.H, self.W), dtype=np.float32)
        label_tensor = np.zeros((self.T, 1, self.H, self.W), dtype=np.float32)

        # Randomly decide whether to draw in top or bottom half
        top_half = np.random.rand() < 0.5
        row_start = 0 if top_half else self.H // 2
        row_end = self.H // 2 if top_half else self.H

        # Randomly choose a column to draw the line
        col = np.random.randint(0, self.W)

        for t in range(self.T):
            x_start = max(0, col - self.line_width // 2)
            x_end = min(self.W, col + self.line_width // 2 + 1)

            # Draw vertical band in either top or bottom half
            label_tensor[t, 0, row_start:row_end, x_start:x_end] = 1.0
            input_tensor[t, 0, row_start:row_end, x_start:x_end] = 1.0
            input_tensor[t, 0] = add_noise(input_tensor[t, 0], chance=self.noise, flip=True)

        input_tensor = np.clip(input_tensor, 0.0, 1.0)
        return torch.from_numpy(input_tensor), torch.from_numpy(label_tensor)
    
    
    def produce_moving_line(self, heavy_noise_prob=0.2, heavy_noise=0.4):
        input_tensor = np.zeros((self.T, 1, self.H, self.W), dtype=np.float32)
        label_tensor = np.zeros((self.T, 1, self.H, self.W), dtype=np.float32)

        top_half = np.random.rand() < 0.5
        row_start = 0 if top_half else self.H // 2
        row_end = self.H // 2 if top_half else self.H

        start_col = np.random.randint(0, self.W - self.W // 4)
        end_col = start_col + self.W // 4

        for t in range(self.T):
            col_pos = int(start_col + (t / self.T) * (end_col - start_col))
            x_start = max(0, col_pos - self.line_width // 2)
            x_end = min(self.W, col_pos + self.line_width // 2 + 1)

            input_tensor[t, 0, row_start:row_end, x_start:x_end] = 1.0

            # Decide whether to corrupt this frame
            is_corrupt = np.random.rand() < heavy_noise_prob
            
            # Noise level based on corruption
            noise_chance = heavy_noise if is_corrupt else self.noise
            input_tensor[t, 0] = add_noise(input_tensor[t, 0], chance=noise_chance, flip=True)

        final_x_start = max(0, end_col - self.line_width // 2)
        final_x_end = min(self.W, end_col + self.line_width // 2 + 1)
        label_tensor[:, 0, row_start:row_end, final_x_start:final_x_end] = 1.0

        input_tensor = np.clip(input_tensor, 0.0, 1.0)
        return torch.from_numpy(input_tensor), torch.from_numpy(label_tensor)
    
    
        
def add_noise(image, chance=0.1, flip=True):
    """
    Add or flip noise pixels in the input tensor.

    Parameters:
    - image: 2D numpy array
    - chance: per-pixel probability of flipping
    - flip: if True, flips pixels (0→1, 1→0); otherwise sets to 1
    """
    mask = np.random.rand(*image.shape) < chance
    if flip:
        image[mask] = 1.0 - image[mask]  # flip 0↔1
    else:
        image[mask] = 1.0
    return image

def build_demo_dataloader(batch_size=4, 
                          time_steps=10, 
                          input_size=(128, 128), 
                          num_workers=0, 
                          num_samples=100,
                          moving=False,
                          noise=0.1,
                          heavy_noise=0.4,
                          heavy_noise_prob=0.3):
    """
    Build a DataLoader for the demo segmentation dataset.
    Args:
        batch_size: Number of samples per batch.
        time_steps: Number of time steps in each sample.
        input_size: Tuple (H, W)
        num_workers: Number of workers for data loading.
        num_samples: Total number of samples in the dataset.
        moving: If True, produces moving lines; otherwise static lines.
    """
    dataset = DemoSegmentationDataset(num_samples=num_samples, 
                                        time_steps=time_steps, 
                                        input_size=input_size,
                                        line_width=3,
                                        moving=moving,
                                        noise=noise,
                                        heavy_noise=heavy_noise,
                                        heavy_noise_prob=heavy_noise_prob)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False)



# ---------------------------
# Visualization utilities
# ---------------------------

def plot_sample_sequence(inputs, labels, sample_idx=0, save_path=None, show=True):
    """
    Plot a sequence of frames (input vs label) for one sample.

    Args:
        inputs: Tensor [B, T, 1, H, W]
        labels: Tensor [B, T, 1, H, W]
        sample_idx: index in batch to visualize
        save_path: if given, saves the image
        show: whether to show the plot
    """
    T = inputs.shape[1]
    fig, axs = plt.subplots(T, 2, figsize=(6, 2 * T))

    for t in range(T):
        axs[t, 0].imshow(inputs[sample_idx, t, 0], cmap='gray', vmin=0, vmax=1)
        axs[t, 0].set_title(f"Input t={t}")
        axs[t, 0].axis("off")

        axs[t, 1].imshow(labels[sample_idx, t, 0], cmap='gray', vmin=0, vmax=1)
        axs[t, 1].set_title(f"Label t={t}")
        axs[t, 1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


# Test visualization
if __name__ == "__main__":
    loader = build_demo_dataloader(input_size=(32, 32), 
                                   time_steps=10,
                                   num_samples=1, 
                                   moving=True,
                                   noise=0.1,
                                   heavy_noise=0.4,
                                   heavy_noise_prob=0.3)
    for x, y in loader:
        print("Input:", x.shape)  
        print("Label:", y.shape)
        plot_sample_sequence(x, y, sample_idx=0)
        break
    
