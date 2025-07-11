from data.demo_loader import build_demo_dataloader
from data.det_loader  import build_det_dataloaders

class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_loaders(self):
        data_cfg = self.cfg.data
        if data_cfg.loader == "demo":
            # exactly your old demo loader logic
            return {
                "train": build_demo_dataloader(
                    batch_size=data_cfg.batch_size,
                    time_steps=data_cfg.time_steps,
                    input_size=tuple(data_cfg.input_size),
                    num_workers=data_cfg.num_workers,
                    num_samples=int(data_cfg.num_samples * 0.8),
                    moving=data_cfg.moving,
                    noise=data_cfg.noise,
                    heavy_noise=data_cfg.heavy_noise,
                    heavy_noise_prob=data_cfg.heavy_noise_prob,
                ),
                "val": build_demo_dataloader(
                    batch_size=data_cfg.batch_size,
                    time_steps=data_cfg.time_steps,
                    input_size=tuple(data_cfg.input_size),
                    num_workers=data_cfg.num_workers,
                    num_samples=int(data_cfg.num_samples * 0.2),
                    moving=data_cfg.moving,
                    noise=data_cfg.noise,
                    heavy_noise=data_cfg.heavy_noise,
                    heavy_noise_prob=data_cfg.heavy_noise_prob,
                ),
                "test": build_demo_dataloader(
                    batch_size=data_cfg.batch_size,
                    time_steps=data_cfg.time_steps,
                    input_size=tuple(data_cfg.input_size),
                    num_workers=data_cfg.num_workers,
                    num_samples=int(data_cfg.num_samples * 0.2),  # or use a separate test set
                    moving=data_cfg.moving,
                    noise=data_cfg.noise,
                    heavy_noise=data_cfg.heavy_noise,
                    heavy_noise_prob=data_cfg.heavy_noise_prob,
                )
            }

        elif data_cfg.loader == "det":
            # use the real DET loader
            # here we simply instantiate train and val on the full dataset
            # (if you want a split, you can wrap the Dataset yourself)
            return build_det_dataloaders(
                batch_size=data_cfg.batch_size,
                num_workers=data_cfg.num_workers,
                downscale_factor=data_cfg.downscale,
                used_T=data_cfg.used_T,
                train_split=0.8  # or expose as cfg parameter
            )

        else:
            raise ValueError(f"Unknown data.loader: {data_cfg.loader}")
