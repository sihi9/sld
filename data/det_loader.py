import os
import h5py
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data.data_utils import apply_label_smoothing
from utils.visualizations import show_sample_triplet

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

def apply_affine_transform(image, M, shape):
    """Apply affine transform to a single-channel image with replicated borders."""
    return cv2.warpAffine(image, M, (shape[1], shape[0]),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE)



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
                 is_test: bool = False,
                 downscale_factor: int = 1,
                 model_downscale: int = None,
                 model_initial_downscale: int = 1,
                 filter_fn=None,
                 used_T = None,
                 use_static=False,
                 use_poisson=False,
                 label_smoothing_enabled=False,
                 smooth_bg=0.05,
                 smooth_lane=0.95,
                 augmentation_intensity=0.1):
        """
        Initialize the dataset.
        Args:            
            h5_path: Path to the HDF5 file.
            is_test: If True, will not apply augmentation.
            downscale_factor: Factor by which to downscale the frames and labels.
            model_downscale: Factor by which the model will downscale the frames.
            model_initial_downscale: Initial downscale factor of the model.
            filter_fn: Optional function to filter samples based on input and label.
            used_T: If specified, only the last `used_T` frames will  be used.
            use_static: If True, will only use last frame.
            use_poisson: If True, will create poisson distribution of last frame
            label_smoothing_enabled: If True, will apply label smoothing.
            smooth_bg: Background label smoothing value.
            smooth_lane: Lane label smoothing value.
            augmentation_intensity: Intensity of random affine transformations.
        """
        path_prefix = './data/DET/'  # Assuming data files are in a 'data' directory
        self.h5_path = path_prefix + h5_path
        self.is_test = is_test
        self.downscale_factor = downscale_factor
        self.model_downscale = model_downscale
        self.model_initial_downscale = model_initial_downscale
        self.filter_fn = filter_fn
        self.used_T = used_T
        self.use_static = use_static
        use_poisson = use_poisson
        self.label_smoothing_enabled = label_smoothing_enabled
        self.smooth_bg = smooth_bg
        self.smooth_lane = smooth_lane
        self.augmentation_intensity = augmentation_intensity

        self.preload = False  # todo: make this configurable
        
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

        if self.preload:
            print(f"Preloading dataset {h5_path} into memory...")
            self._cached_data = []
            for i in range(len(self)):
                self._cached_data.append(self._load_and_process_item(i))
                
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get item by index.
        Args:
            idx: Index of the item to retrieve.
        Returns:
            Tuple of input tensor and label tensor.
        """
        if hasattr(self, "_cached_data"):
            return self._cached_data[idx]
        
        return self._load_and_process_item(idx)

    def _load_and_process_item(self, idx):
        """
        Load and preprocess a single item from the dataset.
        Args:
            idx: Index of the item to load.
        Returns:
            Processed input and label tensors.
        """
        real_idx = self.indices[idx]
        x_np = self._X[real_idx]  # (T,1,H,W)
        y_np = self._Y[real_idx]  # (1,H,W)
        T, C, H, W = x_np.shape

        if self.used_T is not None and self.used_T < T:
            x_np = x_np[-self.used_T:]
            T = self.used_T

        # Process input frames
        frames = []
        if self.use_static:
            img = x_np[-1, 0].astype(np.uint8)
            img_ds = _downscale_frame(img, self.downscale_factor)
            frames = [img_ds for _ in range(T)]
        else:
            for t in range(T):
                img = x_np[t, 0].astype(np.uint8)
                img_ds = _downscale_frame(img, self.downscale_factor)
                frames.append(img_ds)

        x_ds = np.stack(frames, axis=0)  # (T,H2,W2)
        x_ds = x_ds[:, np.newaxis, :, :]  # (T,1,H2,W2)

        # Process label
        lab = (y_np[0] > 0).astype(np.uint8)
        total_label_downscale = self.downscale_factor * self.model_initial_downscale
        lab_ds = _downscale_label(lab, total_label_downscale)
        lab_ds = lab_ds[np.newaxis, :, :]

        # === COMPUTE AUGMENTATION PARAMETERS ===
        if self.augmentation_intensity > 0.0:
            H2, W2 = x_ds.shape[-2], x_ds.shape[-1]
            max_tx = self.augmentation_intensity * W2
            max_angle_rad = np.arcsin(self.augmentation_intensity)
            max_angle_deg = np.degrees(max_angle_rad)

            crop_margin_h = int(H2 * self.augmentation_intensity)
            crop_margin_w = int(W2 * self.augmentation_intensity)
            
            # Make sure label can be croped correctly as well
            crop_margin_h -= crop_margin_h % self.model_initial_downscale
            crop_margin_w -= crop_margin_w % self.model_initial_downscale

            self._max_tx = max_tx
            self._max_angle = max_angle_deg
            self._crop_margin_h = crop_margin_h
            self._crop_margin_w = crop_margin_w
        else:
            self._max_tx = 0
            self._max_angle = 0
            self._crop_margin_h = 0
            self._crop_margin_w = 0

        # === AUGMENTATION ===
        if not self.is_test:
            x_ds, lab_ds = self._augment_sample(x_ds, lab_ds)

        # === FINAL CROP FOR CONSISTENCY EVEN IF NOT AUGMENTED (Test set) ===
        if self.augmentation_intensity > 0.0:
            ch, cw = self._crop_margin_h, self._crop_margin_w
            x_ds = x_ds[:, :, ch:-ch, cw:-cw]
            # compute label crop scaled to its resolution
            ch_lab = ch // self.model_initial_downscale
            cw_lab = cw // self.model_initial_downscale
            lab_ds = lab_ds[:, ch_lab:-ch_lab, cw_lab:-cw_lab]
            
        # === POST-CROP MODEL COMPATIBILITY CHECK ===
        T, C, H2, W2 = x_ds.shape
        rem_h = (H2 % self.model_downscale) if self.model_downscale else 0
        rem_w = (W2 % self.model_downscale) if self.model_downscale else 0

        if rem_h != 0 or rem_w != 0:
            crop_top = rem_h
            crop_left = rem_w // 2
            crop_right = rem_w - crop_left

            x_ds = x_ds[:, :, crop_top:H2, crop_left:W2 - crop_right]

            label_crop_top = crop_top // self.model_initial_downscale
            label_crop_left = crop_left // self.model_initial_downscale
            label_crop_right = crop_right // self.model_initial_downscale
            _, H_lab, W_lab = lab_ds.shape

            lab_ds = lab_ds[:, label_crop_top:H_lab, label_crop_left:W_lab - label_crop_right]

        # === TO TENSOR ===
        x_tensor = torch.from_numpy(x_ds).float() / 255.0
        y_tensor = torch.from_numpy(lab_ds).float()

        if self.label_smoothing_enabled:
            y_tensor = apply_label_smoothing(y_tensor, self.smooth_bg, self.smooth_lane)

        return x_tensor, y_tensor

    def close(self):
        """Close the underlying HDF5 file."""
        self._h5.close()

    def _augment_sample(self, x_seq, label):
        """
        Applies rotation or shift to x_seq and label, taking into account different resolutions.
        Args:
            x_seq: shape (T, 1, H, W)
            label: shape (1, H_lab, W_lab)
        Returns:
            Augmented x_seq and label
        """
        T, _, H, W = x_seq.shape
        _, H_lab, W_lab = label.shape

        # Randomly select transformation
        choice = np.random.choice(['rotate', 'shift', 'none'])

        angle = np.random.uniform(-self._max_angle, self._max_angle) if choice == 'rotate' else 0
        tx = np.random.uniform(-self._max_tx, self._max_tx) if choice == 'shift' else 0

        # Transformation matrix for input frames
        center = (W // 2, H // 2)
        M_frame = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_frame[:, 2] += [tx, 0]

        # Apply to frames
        x_aug = np.zeros_like(x_seq)
        for t in range(T):
            x_aug[t, 0] = cv2.warpAffine(
                x_seq[t, 0], M_frame, (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE
            )

        # === Adjust transformation matrix for label scale
            # Adjust matrix for label space
        scale_w = W_lab / W
        scale_h = H_lab / H

        # Scale center accordingly
        center_lab = (W_lab / 2, H_lab / 2)
        M_label = cv2.getRotationMatrix2D(center_lab, angle, 1.0)
        M_label[:, 2] += [tx * scale_w, 0]  # Scale tx as well

        # Apply to label
        label_aug = np.zeros_like(label)
        label_aug[0] = cv2.warpAffine(
            label[0], M_label, (W_lab, H_lab),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE
        )

        return x_aug, label_aug


    def __del__(self):
        try:
            self.close()
        except:
            pass

class MultiHDF5Dataset(Dataset):
    def __init__(self, 
                 h5_paths, 
                 downscale_factor=1, 
                 model_downscale=None,
                 model_initial_downscale=1,
                 filter_fn=None, 
                 used_T=None,
                 use_static=False,
                 use_poisson=False,
                 label_smoothing_enabled=False,
                 smooth_bg=0.05,
                 smooth_lane=0.95,
                 augmentation_intensity=0.1):
        """
        Dataset that combines multiple HDF5 files.
        Args:
            h5_paths: List of paths to HDF5 files.
            downscale_factor: Factor by which to downscale the frames and labels.
            model_downscale: Factor by which the model will downscale the frames.
            model_initial_downscale: Initial downscale factor of the model.
            filter_fn: Optional function to filter samples based on input and label.
            used_T: If specified, only the last `used_T` frames will be used.
            use_static: If True, will only use last frame.
            use_poisson: If True, will create poisson distribution of last frame
            label_smoothing_enabled: If True, will apply label smoothing.
            smooth_bg: Background label smoothing value.
            smooth_lane: Lane label smoothing value.
            augmentation_intensity: Intensity of random affine transformations.
        """
        self.datasets = [
            HDF5Dataset(h5_path=path,
                        is_test=False,  # Default for multi-dataset so far
                        downscale_factor=downscale_factor,
                        model_downscale=model_downscale,
                        model_initial_downscale=model_initial_downscale,
                        filter_fn=filter_fn,
                        used_T=used_T,
                        use_static=use_static,
                        use_poisson=use_poisson,
                        label_smoothing_enabled=label_smoothing_enabled,
                        smooth_bg=smooth_bg,
                        smooth_lane=smooth_lane,
                        augmentation_intensity=augmentation_intensity)
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
                          model_downscale=8,
                          model_initial_downscale=1,
                          used_T=None,
                          use_static=False,
                          use_poisson=False,
                          label_smoothing_enabled=False,
                          smooth_bg=0.05,
                          smooth_lane=0.95,
                          augmentation_intesity=0.1,
                          train_split=0.8,
                          seed=42,
                          shuffle=True,
                          test_file='20190217_1156_T30_x4.h5',
                          data_dir='./data/DET/'):
    """
    Build DataLoader for DET dataset.
    Args:
        batch_size: Batch size for DataLoader.
        num_workers: Number of workers for DataLoader.
        downscale_factor: Factor by which to downscale the frames and labels.
        used_T: If specified, only the last `used_T` frames will be used.
        use_static: If True, will only use last frame.
        use_poisson: If True, will create poisson distribution of last frame.
        label_smoothing_enabled: If True, will apply label smoothing.
        smooth_bg: Background label smoothing value.
        smooth_lane: Lane label smoothing value.
        augmentation_intensity: Intensity of random affine transformations.
        train_split: Fraction of data to use for training (0.8 means 80% train, 20% val).
        seed: Random seed for reproducibility.
        shuffle: Whether to shuffle the training data.
        test_file: Name of the file used for testing.
        data_dir: Directory where the data files are located.
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders.
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
    train_val_files = [f for f in all_files if f != test_file]

    # Build dataset for training and validation
    dataset = MultiHDF5Dataset(
        h5_paths=train_val_files,
        downscale_factor=downscale_factor,
        model_downscale=model_downscale,
        model_initial_downscale=model_initial_downscale,
        used_T=used_T,
        use_static=use_static,
        use_poisson=use_poisson,
        label_smoothing_enabled=label_smoothing_enabled,
        smooth_bg=smooth_bg,
        smooth_lane=smooth_lane,
        augmentation_intensity=augmentation_intesity
    )

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Build dataset for testing
    test_dataset = HDF5Dataset(
        h5_path=test_file,
        is_test=True,  # Test dataset does not use augmentation
        downscale_factor=downscale_factor,
        model_downscale=model_downscale,
        model_initial_downscale=model_initial_downscale,
        used_T=used_T,
        use_static=use_static,
        use_poisson=use_poisson,
        label_smoothing_enabled=label_smoothing_enabled,
        smooth_bg=smooth_bg,
        smooth_lane=smooth_lane,
        augmentation_intensity=augmentation_intesity
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }


def plot_sample_sequence(inputs, labels, history=10, save_path=None, show=True):
    """
    Visualize a sequence of frames with label in last frame
    Args:
        inputs: [B, T, C, H, W] 
        labels: [B, 1, H, W]
        history: Number of frames to show from the end of the sequence
        save_path: Path to save the figure, if None, will not save
        show: If True, will show the figure, otherwise will just close it
    """
    # Squeeze channel dimension
    x_seq = inputs[0, :, 0, :, :]   # [T, H_in, W_in]
    y = labels[0, 0, :, :]          # [H_lab, W_lab]

    # Resize label to match input shape if necessary
    H_in, W_in = x_seq.shape[1:]
    H_lab, W_lab = y.shape
    if (H_in != H_lab) or (W_in != W_lab):
        y_img = Image.fromarray(y.cpu().numpy().astype(np.uint8))
        y_resized = y_img.resize((W_in, H_in), resample=Image.NEAREST)
        y = np.array(y_resized)

    # Create boolean mask for label overlay
    mask = (y != 0)

    T = x_seq.shape[0]
    if T > history:
        seq = x_seq[-history:]
    else:
        seq = x_seq
    n = seq.shape[0]

    # Dynamically adjust figure size
    fig_width = max(n * 2.5, 8)
    fig_height = 4
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
        
def test_time():
    import time
    loader = build_det_dataloaders(num_workers=4, downscale_factor=4, shuffle=False)["train"]

    start = time.time()
    for i, (x, y) in enumerate(loader):
        print(f"Batch {i}: Input {x.shape}, Label {y.shape}")
        break
    print("Time to load 1 batch:", time.time() - start)
    
# Example usage guard
if __name__ == '__main__':
    # Quick test
    # test_time()
    loaders = build_det_dataloaders(
        downscale_factor=1,
        model_initial_downscale=4,
        model_downscale=4, 
        shuffle=False, 
        augmentation_intesity=0.05
    )

    # Full dataset sizes (independent of batch size)
    print("Dataset sizes:")
    print("  Train:", len(loaders["train"].dataset))
    print("  Val:  ", len(loaders["val"].dataset))
    print("  Test: ", len(loaders["test"].dataset))

    # Optional quick sanity check on a single batch
    for x, y in loaders["train"]:
        print("Input:", x.shape)
        print("Label:", y.shape)
        plot_sample_sequence(x, y, history=1, save_path='./sample_sequence.png', show=False)
        break
