from data.demo_loader import build_demo_dataloader


class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_loaders(self):
        return {
            "train": build_demo_dataloader(
                batch_size=self.cfg.data.batch_size,
                time_steps=self.cfg.data.time_steps,
                input_size=tuple(self.cfg.data.input_size),
                num_samples=int(self.cfg.data.num_samples * 0.8),
                moving=self.cfg.data.moving,
                noise=self.cfg.data.noise,
                heavy_noise=self.cfg.data.heavy_noise,
                heavy_noise_prob=self.cfg.data.heavy_noise_prob,
            ),
            "val": build_demo_dataloader(
                batch_size=self.cfg.data.batch_size,
                time_steps=self.cfg.data.time_steps,
                input_size=tuple(self.cfg.data.input_size),
                num_samples=int(self.cfg.data.num_samples * 0.2),
                moving=self.cfg.data.moving,
                noise=self.cfg.data.noise,
                heavy_noise=self.cfg.data.heavy_noise,
                heavy_noise_prob=self.cfg.data.heavy_noise_prob,
            ),
        }
