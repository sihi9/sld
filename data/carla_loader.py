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

def _downscale_frame(img: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return img
    H, W = img.shape
    H2, W2 = H // factor, W // factor
    img_ds = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
    _, img_bin = cv2.threshold(img_ds, 255//factor, 255, cv2.THRESH_BINARY)
    return img_bin

def _downscale_label(img: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return img
    H, W = img.shape
    H2, W2 = H // factor, W // factor
    downscaled = cv2.resize(img * factor, (W2, H2), interpolation=cv2.INTER_AREA)
    return (downscaled > 0.05).astype(np.uint8)

class HDF5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        is_test: bool = False,
        downscale_factor: int = 1,
        model_downscale: int = None,
        model_initial_downscale: int = 1,
        filter_fn=None,
        used_T = None,
        use_static=False,
        label_smoothing_enabled=False,
        smooth_bg=0.05,
        smooth_lane=0.95
    ):
        self.h5_path = h5_path
        self.is_test = is_test
        self.downscale_factor = downscale_factor
        self.model_downscale = model_downscale
        self.model_initial_downscale = model_initial_downscale
        self.filter_fn = filter_fn
        self.used_T = used_T
        self.use_static = use_static
        self.label_smoothing_enabled = label_smoothing_enabled
        self.smooth_bg = smooth_bg
        self.smooth_lane = smooth_lane

        self._h5 = h5py.File(self.h5_path, 'r')
        self._X = self._h5['X']
        self._Y = self._h5['Y']
        #self.indices = list(range(self._X.shape[0]))
        self.indices = list(range(0, self._X.shape[0], 15))
        
        if self.filter_fn is not None:
            valid = []
            for idx in self.indices:
                x_np = self._X[idx]
                y_np = self._Y[idx]
                if self.filter_fn(x_np, y_np):
                    valid.append(idx)
            self.indices = valid

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_np = self._X[real_idx]
        y_np = self._Y[real_idx]
        T, C, H, W = x_np.shape

        if self.used_T is not None and self.used_T < T:
            x_np = x_np[-self.used_T:]
            T = self.used_T

        # Downscale input
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

        x_ds = np.stack(frames, axis=0)
        x_ds = x_ds[:, np.newaxis, :, :]  # (T,1,H2,W2)

        # Multi-stage downscale for label
        lab = (y_np[0] > 0).astype(np.uint8)
        total_label_downscale = self.downscale_factor * self.model_initial_downscale
        lab_ds = _downscale_label(lab, total_label_downscale)
        lab_ds = lab_ds[np.newaxis, :, :]  # (1,H2,W2)

        # Model compatibility crop (for arbitrary downscale)
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

        # Convert to torch.Tensor
        x_tensor = torch.from_numpy(x_ds).float() / 255.0
        y_tensor = torch.from_numpy(lab_ds).float()

        # Label smoothing if enabled
        if self.label_smoothing_enabled:
            y_tensor = apply_label_smoothing(y_tensor, self.smooth_bg, self.smooth_lane)

        return x_tensor, y_tensor

    def close(self):
        self._h5.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass

class MultiHDF5Dataset(Dataset):
    def __init__(
        self,
        h5_paths,
        downscale_factor=1,
        model_downscale=None,
        model_initial_downscale=1,
        filter_fn=None,
        used_T=None,
        use_static=False,
        label_smoothing_enabled=False,
        smooth_bg=0.05,
        smooth_lane=0.95
    ):
        self.datasets = [
            HDF5Dataset(
                h5_path=path,
                is_test=False,
                downscale_factor=downscale_factor,
                model_downscale=model_downscale,
                model_initial_downscale=model_initial_downscale,
                filter_fn=filter_fn,
                used_T=used_T,
                use_static=use_static,
                label_smoothing_enabled=label_smoothing_enabled,
                smooth_bg=smooth_bg,
                smooth_lane=smooth_lane,
            ) for path in h5_paths
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
            
def build_carla_dataloaders(
    batch_size=4,
    num_workers=0,
    downscale_factor=1,
    model_downscale=8,
    model_initial_downscale=1,
    used_T=None,
    use_static=False,
    label_smoothing_enabled=False,
    smooth_bg=0.05,
    smooth_lane=0.95,
    shuffle=True,
    data_dir='./data/CARLA/'
):
    def file_from_key(key):
        return f"dataset_{key}_67fps_T30_x2.h5"

    train_keys = ['Town03_4000', 'Town04_6000', 'Town05_10000', 'Town06_10000', 'Town10HD_10000']
    val_keys = ['Town04_5000', 'Town06_5000']
    test_keys = ['Town04_5000']  # you could make this a list for multi-file test

    train_files = [file_from_key(k) for k in train_keys]
    val_files = [file_from_key(k) for k in val_keys]
    test_files = [file_from_key(k) for k in test_keys]

    train_paths = [os.path.join(data_dir, f) for f in train_files]
    val_paths = [os.path.join(data_dir, f) for f in val_files]
    test_paths = [os.path.join(data_dir, f) for f in test_files]

    train_dataset = MultiHDF5Dataset(
        h5_paths=train_paths,
        downscale_factor=downscale_factor,
        model_downscale=model_downscale,
        model_initial_downscale=model_initial_downscale,
        used_T=used_T,
        use_static=use_static,
        label_smoothing_enabled=label_smoothing_enabled,
        smooth_bg=smooth_bg,
        smooth_lane=smooth_lane,
    )

    val_dataset = MultiHDF5Dataset(
        h5_paths=val_paths,
        downscale_factor=downscale_factor,
        model_downscale=model_downscale,
        model_initial_downscale=model_initial_downscale,
        used_T=used_T,
        use_static=use_static,
        label_smoothing_enabled=label_smoothing_enabled,
        smooth_bg=smooth_bg,
        smooth_lane=smooth_lane,
    )

    test_dataset = MultiHDF5Dataset(
        h5_paths=test_paths,
        downscale_factor=downscale_factor,
        model_downscale=model_downscale,
        model_initial_downscale=model_initial_downscale,
        used_T=used_T,
        use_static=use_static,
        label_smoothing_enabled=label_smoothing_enabled,
        smooth_bg=smooth_bg,
        smooth_lane=smooth_lane,
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
        inputs: [B, T, C, H, W] (torch.Tensor or np.ndarray)
        labels: [B, 1, H, W]
        history: Number of frames to show from the end of the sequence
        save_path: Path to save the figure, if None, will not save
        show: If True, will show the figure, otherwise will just close it
    """
    # Support both torch and numpy input
    if isinstance(inputs, torch.Tensor):
        x_seq = inputs[0, :, 0].cpu().numpy()
    else:
        x_seq = inputs[0, :, 0]
    if isinstance(labels, torch.Tensor):
        y = labels[0, 0].cpu().numpy()
    else:
        y = labels[0, 0]
    # Resize label if needed
    H_in, W_in = x_seq.shape[1:]
    H_lab, W_lab = y.shape
    if (H_in != H_lab) or (W_in != W_lab):
        y_img = Image.fromarray(y.astype(np.uint8))
        y_resized = y_img.resize((W_in, H_in), resample=Image.NEAREST)
        y = np.array(y_resized)
    mask = (y != 0)
    T = x_seq.shape[0]
    if T > history:
        seq = x_seq[-history:]
    else:
        seq = x_seq
    n = seq.shape[0]
    fig_width = max(n * 2.5, 8)
    fig_height = 4
    width_ratios = [1] * (n - 1) + [2]
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, n, width_ratios=width_ratios, wspace=0.05, hspace=0)
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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

def test_time():
    import time
    loader = build_carla_dataloaders(num_workers=2, downscale_factor=4, shuffle=False)["train"]
    start = time.time()
    iterations = 500
    for i, (x, y) in enumerate(loader):
        if i >= iterations:
            break
    end = time.time()
    elapsed = end - start
    print(f"Time for {iterations} batches: {elapsed:.2f} seconds")
    print(f"Average time per batch: {elapsed / iterations:.2f} seconds")

if __name__ == '__main__':
    data_type = 'train'  # 'train', 'val', 'test'
    loader = build_carla_dataloaders(downscale_factor=1, 
                                     model_downscale=1,
                                     model_initial_downscale=1, shuffle=True)[data_type]
    for i, (x, y) in enumerate(loader):
        print("Input:", x.shape)
        print("Label:", y.shape)
        plot_sample_sequence(x, y, history=1, save_path=f'./{data_type}_sample_sequence{i}.png', show=False)
        if i >= 5:
            break
