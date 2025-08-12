from data.demo_loader import build_demo_dataloader
from data.det_loader  import build_det_dataloaders
from data.carla_loader import build_carla_dataloaders

class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_loaders(self):
        data_cfg = self.cfg.data
        
        # caluclate model downscale factor so dataloader can prepare data accordingly
        initial_scaling = self.cfg.model.initial_scaling if hasattr(self.cfg.model, 'initial_scaling') else 1
        model_downscale = 2 ** (len(self.cfg.model.features) - 1) if hasattr(self.cfg.model, 'features') else 1
        total_model_downscale = initial_scaling * model_downscale
        
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
            return build_det_dataloaders(
                batch_size=data_cfg.batch_size,
                num_workers=data_cfg.num_workers,
                downscale_factor=data_cfg.downscale,
                model_downscale=total_model_downscale,
                model_initial_downscale=initial_scaling,
                used_T=data_cfg.used_T,
                use_static=data_cfg.use_static,
                label_smoothing_enabled=data_cfg.label_smoothing.enabled,
                smooth_bg=data_cfg.label_smoothing.background,
                smooth_lane=data_cfg.label_smoothing.lane,
                augmentation_intesity=data_cfg.augmentation_intensity,
                train_split=0.8 
            )
        elif data_cfg.loader == "carla":
            # use the real CARLA loader
            return build_carla_dataloaders(
                batch_size=data_cfg.batch_size,
                num_workers=data_cfg.num_workers,
                downscale_factor=data_cfg.downscale,
                model_downscale=total_model_downscale,
                model_initial_downscale=initial_scaling,
                used_T=data_cfg.used_T,
                use_static=data_cfg.use_static,
                label_smoothing_enabled=data_cfg.label_smoothing.enabled,
                smooth_bg=data_cfg.label_smoothing.background,
                smooth_lane=data_cfg.label_smoothing.lane,
            )

        else:
            raise ValueError(f"Unknown data.loader: {data_cfg.loader}")

