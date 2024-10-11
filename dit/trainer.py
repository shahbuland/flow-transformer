import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
import os
from dataclasses import asdict
from ema_pytorch import EMA

from .configs import TrainConfig, LoggingConfig, ModelConfig
from .utils import get_scheduler_cls, Stopwatch, get_extra_optimizer
from .sampling import Sampler, CFGSampler, to_wandb_batch

class Trainer:
    def __init__(self, config : TrainConfig, logging_config : LoggingConfig = None, model_config : ModelConfig = None):
        self.config = config
        self.logging_config = logging_config
        self.model_config = model_config

        self.accum_steps = self.config.target_batch_size // self.config.batch_size
        self.accelerator = Accelerator(
            log_with = "wandb",
            gradient_accumulation_steps = self.accum_steps
        )

        tracker_kwargs = {}
        if self.logging_config is not None:
            log = self.logging_config
            tracker_kwargs['wandb'] = {
                'name' : log.run_name,
                'entity' : log.wandb_entity,
                'mode' : 'online'
            }

            config_dict = asdict(config)
            if model_config is not None:
                config_dict.update(asdict(model_config))


            self.accelerator.init_trackers(
                project_name = log.wandb_project,
                config = config_dict,
                init_kwargs = tracker_kwargs
            )

        self.world_size = self.accelerator.state.num_processes
        self.total_step_counter = 0

        self.ema = None

    def get_should(self, step = None):
        # Get a dict of bools that determines if certain things should be done at the current step
        if step is None:
            step = self.total_step_counter
        return {
            "log" : step % self.config.log_interval == 0 and self.accelerator.sync_gradients,
            "save" : step % self.config.save_interval == 0 and self.accelerator.sync_gradients,
            "sample" : step % self.config.sample_interval == 0 and self.accelerator.sync_gradients
        }

    def save(self, step = None, dir = None):
        """
        In directory, save checkpoint of accelerator state using step and self.logging_config.run_name
        """
        if step is None:
            step = self.total_step_counter
        if dir is None:
            dir = os.path.join(self.config.checkpoint_root_dir, f"{self.logging_config.run_name}_{step}")
        
        os.makedirs(dir, exist_ok = True)

        self.accelerator.save_state(output_dir = dir)
        if self.ema is not None:
            ema_path = os.path.join(dir, "ema_model.pth")
            torch.save(self.ema.state_dict(), ema_path)
            ema_model_path = os.path.join(dir, "out.pth")
            torch.save(self.ema.ema_model.state_dict(), ema_model_path)

    def train(self, model, loader):
        try:
            opt_class = getattr(torch.optim, self.config.opt)
        except:
            opt_class = get_extra_optimizer(self.config.opt)
        opt = opt_class(model.parameters(), **self.config.opt_kwargs)

        if self.logging_config is not None:
            wandb.watch(model)

        scheduler = None
        if self.config.scheduler is not None:
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler)
            except:
                scheduler_class = get_scheduler_cls(self.config.scheduler)
            scheduler = scheduler_class(opt, **self.config.scheduler_kwargs)

        if scheduler:
            model, loader, opt, scheduler = self.accelerator.prepare(model, loader, opt, scheduler)
        else:
            model, loader, opt = self.accelerator.prepare(model, loader, opt)

        self.ema = EMA(
            self.accelerator.unwrap_model(model),
            beta = 0.9999,
            update_after_step = 100,
            update_every = 10
        )

        sw = Stopwatch()
        sw.reset()

        if self.model_config.cfg_prob > 0.0:
            sampler = CFGSampler()
        else:
            sampler = Sampler()
        
        for epoch in range(self.config.epochs):
            for i, batch in enumerate(loader):
                with self.accelerator.accumulate(model):
                    loss = model(batch)
                    
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if self.config.grad_clip > 0: self.accelerator.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                        self.total_step_counter += 1
                        self.ema.update()

                    opt.step()
                    if scheduler:
                        scheduler.step()
                    opt.zero_grad()

                    should = self.get_should()
                    if self.logging_config is not None and should['log'] or should['sample']:
                        wandb_dict = {
                            "loss": loss.item(),
                            "time_per_1k" : sw.hit(self.config.target_batch_size),
                        }
                        if scheduler:
                            wandb_dict["learning_rate"] = scheduler.get_last_lr()[0]
                        if should['sample']:
                            n_samples = self.config.n_samples
                            images = to_wandb_batch(sampler.sample(n_samples, self.ema.ema_model, self.config.sample_prompts))
                            wandb_dict.update({
                                "samples": images
                            })
                        wandb.log(wandb_dict)
                        sw.reset()
                    if should['save']:
                        self.save(self.total_step_counter)