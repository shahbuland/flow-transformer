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
from .validation import Validator, PickScorer

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

        def should_fn(interval):
            return step % interval == 0 and self.accelerator.sync_gradients

        return {
            "log" : should_fn(self.config.log_interval),
            "save" : should_fn(self.config.save_interval),
            "sample" : should_fn(self.config.sample_interval),
            "val" : should_fn(self.config.val_interval)
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
    
    def load(self):
        """
        Load the latest checkpoint from the checkpoint directory.
        """
        checkpoint_dir = self.config.checkpoint_root_dir
        run_name = self.logging_config.run_name
        
        # Get all directories that match the run name pattern
        matching_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith(run_name) and d[len(run_name):].strip('_').isdigit()]
        
        if not matching_dirs:
            print(f"No checkpoints found for {run_name}")
            _ = input("Are you sure you want to continue?")
            return
        
        # Sort directories by the number at the end and get the latest
        latest_dir = max(matching_dirs, key=lambda x: int(x[len(run_name):].strip('_')))
        latest_checkpoint = os.path.join(checkpoint_dir, latest_dir)
        
        print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Load accelerator state
        self.accelerator.load_state(latest_checkpoint)
        
        # Load EMA if it exists
        ema_path = os.path.join(latest_checkpoint, "ema_model.pth")
        if os.path.exists(ema_path) and self.ema is not None:
            self.ema.load_state_dict(torch.load(ema_path))
        
        print("Checkpoint loaded successfully")

    def train(self, model, loader, val_loader = None):
        # optimizer setup 
        try:
            opt_class = getattr(torch.optim, self.config.opt)
        except:
            opt_class = get_extra_optimizer(self.config.opt)
        opt = opt_class(model.parameters(), **self.config.opt_kwargs)

        # scheduler setup
        scheduler = None
        if self.config.scheduler is not None:
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler)
            except:
                scheduler_class = get_scheduler_cls(self.config.scheduler)
            scheduler = scheduler_class(opt, **self.config.scheduler_kwargs)

        # accelerator prepare
        model.train()
        if scheduler:
            model, loader, opt, scheduler = self.accelerator.prepare(model, loader, opt, scheduler)
        else:
            model, loader, opt = self.accelerator.prepare(model, loader, opt)

        # ema setup
        self.ema = EMA(
            self.accelerator.unwrap_model(model),
            beta = 0.9999,
            update_after_step = 100,
            update_every = 1,
            ignore_names = {'repa', 'vae', 'text_embedder'},
            coerce_dtype = True
        )

        # load checkpoint if we want to 
        if self.config.resume:
            self.load()

        sw = Stopwatch()
        sw.reset()

        if self.logging_config is not None:
            wandb.watch(self.accelerator.unwrap_model(model), log = 'all')

        # Set up sampler (maybe with cfg)
        if self.model_config.cfg_prob > 0.0:
            sampler = CFGSampler()
        else:
            sampler = Sampler()

        # Set up validator
        validator = None
        if val_loader is not None:
            validator = Validator(self.accelerator.prepare(val_loader), self.config.batch_size * self.config.val_batch_mult)
        scorer = PickScorer(self.config.batch_size * self.config.val_batch_mult)

        for epoch in range(self.config.epochs):
            for i, batch in enumerate(loader):
                with self.accelerator.accumulate(model):
                    loss, extra = model(batch)
                    
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        if self.config.grad_clip > 0: self.accelerator.clip_grad_norm_(model.parameters(), self.config.grad_clip)

                    opt.step()
                    if scheduler:
                        scheduler.step()
                    opt.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.total_step_counter += 1
                        self.accelerator.unwrap_model(model).normalize()
                        self.ema.update()

                    should = self.get_should()
                    if self.logging_config is not None and should['log'] or should['sample']:
                        wandb_dict = {
                            "loss": extra['diff_loss'].item(),
                            "time_per_1k" : sw.hit(self.config.target_batch_size),
                            "last_hidden_min": extra['last_hidden'].min(),
                            "last_hidden_max": extra['last_hidden'].max(),
                            "last_hidden_mean": extra['last_hidden'].mean()
                        }
                        if self.model_config.repa_weight > 0:
                            wandb_dict['repa_loss'] = extra['repa_loss'].item()
                        if scheduler:
                            wandb_dict["learning_rate"] = scheduler.get_last_lr()[0]
                        if should['sample']:
                            n_samples = self.config.n_samples
                            images = to_wandb_batch(
                                sampler.sample(n_samples, self.ema.ema_model, self.config.sample_prompts),
                                self.config.sample_prompts
                            )
                            wandb_dict.update({
                                "samples": images
                            })
                        wandb.log(wandb_dict)
                        sw.reset()
                    if should['save']:
                        self.save(self.total_step_counter)
                    if should['val'] and validator is not None:
                        self.ema.ema_model.eval()
                        val_loss = validator(self.ema.ema_model)
                        pick_score = scorer(sampler, self.ema.ema_model)
                        self.ema.ema_model.train()
                        
                        wandb.log({
                            'validation_loss' : val_loss,
                            'pick_score' : pick_score
                        })
                        
