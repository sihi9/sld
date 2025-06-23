import os
import h5py
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _downscale_frame(img: np.ndarray, factor: int) -> np.ndarray:
    """Downscale binary frame (0/255) with area then threshold to preserve binary."""
    if factor == 1:
        return img
    H, W = img.shape
    H2, W2 = H // factor, W // factor
    img_ds = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
    # re-binarize (removes isolated noise)
    _, img_bin = cv2.threshold(img_ds, 255//factor, 255, cv2.THRESH_BINARY)
    return img_bin


def _downscale_label(img: np.ndarray, factor: int) -> np.ndarray:
    """Downscale integer-labelled mask with nearest-neighbor interpolation."""
    if factor == 1:
        return img
    
   
    H, W = img.shape
    H2, W2 = H // factor, W // factor
    # Problem are disappearing labels, workaround is to use area interpolation
    #return cv2.resize(img, (W2, H2), interpolation=cv2.INTER_NEAREST)
    
    downscaled = cv2.resize(img * factor, (W2, H2), interpolation=cv2.INTER_AREA)
    # Now threshold — retain block if any lane pixels were present
    return (downscaled > 0.05).astype(np.uint8)  # OR use >0.05 to be stricter


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset for HDF5 sequence data.

    Args:
        h5_path: Path to the HDF5 file containing 'X', 'Y', and optionally 'snapshot_idx'.
        downscale_factor: Factor by which to downscale the frames and labels.
        filter_fn: Optional function (X: np.ndarray, Y: np.ndarray) -> bool
                   to include/exclude samples.
    """
    def __init__(self,
                 h5_path: str,
                 downscale_factor: int = 1,
                 filter_fn=None,
                 used_T = None):
        path_prefix = './data/DET/'  # Assuming data files are in a 'data' directory
        self.h5_path = path_prefix + h5_path
        self.downscale_factor = downscale_factor
        self.filter_fn = filter_fn
        self.used_T = used_T
        
        # Open in read-only mode
        self._h5 = h5py.File(self.h5_path, 'r')
        self._X = self._h5['X']
        self._Y = self._h5['Y']
        self._snapshot_idx = self._h5['snapshot_idx']
        # Build index list
        self.indices = list(range(self._X.shape[0]))

        # Apply filter_fn if provided
        if self.filter_fn is not None:
            valid = []
            for idx in self.indices:
                x_np = self._X[idx]  # shape (T,1,H,W)
                y_np = self._Y[idx]
                # numpy array
                if self.filter_fn(x_np, y_np):
                    valid.append(idx)
            self.indices = valid

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_np = self._X[real_idx]  # (T,1,H,W)
        y_np = self._Y[real_idx]  # (1,H,W)
        # Downscale each frame and label
        # Convert to uint8 numpy
        T, C, H, W = x_np.shape
        
        if self.used_T is not None and self.used_T < T:
            x_np = x_np[-self.used_T:]  # Keep last `used_T` frames
            T = self.used_T
    
        # Process frames
        
        frames = []
        for t in range(T):
            img = x_np[t, 0, :, :].astype(np.uint8)
            img_ds = _downscale_frame(img, self.downscale_factor)
            frames.append(img_ds)
        x_ds = np.stack(frames, axis=0)  # (T,H2,W2)
        x_ds = x_ds[:, np.newaxis, :, :]  # (T,1,H2,W2)

        # Process label
        lab = (y_np[0] > 0).astype(np.uint8)    # make binary
        lab_ds = _downscale_label(lab, self.downscale_factor)
        lab_ds = lab_ds[np.newaxis, :, :]  # (1,H2,W2)

        # make sure the image can be fed into a U-Net model with 3 2x2 downscales
        T, C, H2, W2 = x_ds.shape
        rem_h = H2 % 8
        rem_w = W2 % 8
        
        if rem_h != 0 or rem_w != 0:
            # Crop the top because its less relevant
            crop_top = rem_h
            
            # split width cropping
            crop_left  = rem_w // 2
            crop_right = rem_w - crop_left

            # apply to both data and label
            x_ds = x_ds[:, :,
                        crop_top : H2,
                        crop_left: W2 - crop_right]
            lab_ds = lab_ds[:,
                            crop_top : H2,
                            crop_left: W2 - crop_right]
            
        
        # Convert to torch.Tensor
        x_tensor = torch.from_numpy(x_ds).float() / 255.0
        y_tensor = torch.from_numpy(lab_ds).float()
        
        return x_tensor, y_tensor

    def close(self):
        """Close the underlying HDF5 file."""
        self._h5.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass

class MultiHDF5Dataset(Dataset):
    def __init__(self, h5_paths, downscale_factor=1, filter_fn=None, used_T=None):
        self.datasets = [
            HDF5Dataset(h5_path=path,
                        downscale_factor=downscale_factor,
                        filter_fn=filter_fn,
                        used_T=used_T)
            for path in h5_paths
        ]
        self.cumulative_lengths = np.cumsum([len(ds) for ds in self.datasets])
        
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        local_idx = idx if dataset_idx == 0 else idx - self.cumulative_lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][local_idx]
    
    def close(self):
        for ds in self.datasets:
            ds.close()
            
def build_det_dataloaders(batch_size=4, 
                          num_workers=0, 
                          downscale_factor=1,
                          used_T=None,
                          train_split=0.8,
                          seed=42,
                          shuffle=True,
                          test_file='20190222_1707_T30_x4.h5',
                          data_dir='./data/DET/'):
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    train_val_files = [f for f in all_files if f != test_file]

    # Build dataset for training and validation
    dataset = MultiHDF5Dataset(
        h5_paths=train_val_files,
        downscale_factor=downscale_factor,
        used_T=used_T
    )

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Build dataset for testing
    test_dataset = HDF5Dataset(
        h5_path=test_file,
        downscale_factor=downscale_factor,
        used_T=used_T
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

def plot_sample_sequence(inputs, labels, history=10, save_path=None, show=True):
    """
    Visualize a sequence of frames with label in last frame
    Args:
        inputs: [B, T, C, H, W] 
        labes: [B, 1, H, W]
        history: Number of frames to show from the end of the sequence
        save_path: Path to save the figure, if None, will not save
        show: If True, will show the figure, otherwise will just close it
    """

    # Squeeze channel dimension
     
    x_seq = inputs[0, :, 0, :, :]   
    y = labels[0, 0, :, :]
    # Create boolean mask for label overlay
    mask = (y != 0)

    T = x_seq.shape[0]

    
    if T > history:
        seq = x_seq[-history:]
    else:
        seq = x_seq
    n = seq.shape[0]

    # Dynamically adjust figure size
    # Each frame column approx 2 inches, but minimum width 6 inches
    fig_width = max(n * 2.5, 8)
    fig_height = 4
    # Set up figure: single row with n columns, last column wider
    width_ratios = [1] * (n - 1) + [2]
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        1, n,
        width_ratios=width_ratios,
        wspace=0.05, hspace=0)

    # Plot frames and overlay mask on last frame
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(seq[i], cmap='gray', vmin=0, vmax=1)
        if i == n - 1:
            # Create RGBA overlay: transparent background, red where mask True
            H, W = y.shape
            overlay = np.zeros((H, W, 4), dtype=float)
            overlay[mask, 0] = 1.0  # red channel
            overlay[mask, 3] = 0.4  # alpha channel
            ax.imshow(overlay)
        ax.axis('off')

    # Save figure
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
        
# Example usage guard
if __name__ == '__main__':
    # Quick test
    loader = build_det_dataloaders(downscale_factor=1, shuffle=False)["train"]
    
    for x, y in loader:
        print("Input:", x.shape)  
        print("Label:", y.shape)
        plot_sample_sequence(x, y, history=1)
        break
