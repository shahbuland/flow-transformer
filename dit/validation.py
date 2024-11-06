import torch
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from PIL import Image

from .data import create_loader

class Validator:
    def __init__(self, validation_loader, val_batch_size : int, total_size = 10000):
        self.loader = validation_loader
        self.total_size = total_size
        self.b_size = val_batch_size

    @torch.no_grad()
    def __call__(self, model):
        total_loss = 0.
        n_samples = 0
        print("Validating...")
        for batch in tqdm(self.loader, total=self.total_size // self.b_size):
            loss, extra = model(batch)
            total_loss += loss.item()

            n_samples += self.b_size
            if n_samples >= self.total_size:
                break

        return loss.item() / self.total_size

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize, ToTensor
import os
from tqdm import tqdm
import joblib

class FIDScorer:
    def __init__(self, validation_loader, cache_path='./fid_cache.pkl', total_size=10000, batch_size=256, device='cuda', n_sampling_steps : int = 32):
        self.loader = validation_loader
        self.total_size = total_size
        self.batch_size = batch_size
        self.device = device
        self.cache_path = cache_path
        self.n_sampling_steps = n_sampling_steps
        self.fid = FrechetInceptionDistance(feature=2048).to(device)

        if os.path.exists(self.cache_path):
            self.load(self.cache_path)
        else:
            self._compute_real_stats()

    def load(self, path):
        real_stats = joblib.load(path)
        real_f_sum, real_f_cov_sum, real_f_n = real_stats
        self.fid.real_features_sum = real_f_sum
        self.fid.real_features_cov_sum = real_f_cov_sum
        self.fid.real_features_num_samples = real_f_n
        print(f"Loaded FID statistics for {real_f_n} real images from {path}")

    def save(self, path):
        real_f_sum = self.fid.real_features_sum
        real_f_cov_sum = self.fid.real_features_cov_sum
        real_f_n = self.fid.real_features_num_samples
        joblib.dump([real_f_sum, real_f_cov_sum, real_f_n], path)

    @torch.no_grad()
    def _compute_real_stats(self):
        print("Computing FID statistics for real images...")
        processed_samples = 0
        all_images = []
        self.fid.reset()
        for images, _ in tqdm(self.loader):
            if processed_samples >= self.total_size:
                break
            images = (images * 255).byte().to(self.device)
            self.fid.update(images, real = True)
            processed_samples += images.shape[0]
        
        print(f"Processed {processed_samples} real images for FID calculation.")
        self.save(self.cache_path)

    @torch.no_grad()
    def __call__(self, sampler, model):
        self.fid.reset()
        self.load(self.cache_path)

        print("Generating images for FID calculation...")
        prompts = []
        for _, batch_prompts in self.loader:
            prompts.extend(batch_prompts)
            if len(prompts) >= self.total_size:
                prompts = prompts[:self.total_size]
                break

        print(f"Total prompts: {len(prompts)}")

        all_images = []

        old_steps = sampler.config.n_steps
        sampler.config.n_steps = self.n_sampling_steps

        for i in tqdm(range(0, self.total_size, self.batch_size)):
            batch_size = min(self.batch_size, self.total_size - i)
            batch_prompts = prompts[i:i+batch_size]
            
            images = sampler.sample(batch_size, model, batch_prompts)
            images = (images * 255).byte().to(self.device)
            self.fid.update(images, real = False)
        
        sampler.config.n_steps = old_steps
           
        return self.fid.compute().item()


class PickScorer:
    def __init__(self, batch_size : int = 256, n_samples : int = None, device = 'cuda', n_sampling_steps : int = 32):
        self.n_sampling_steps = n_sampling_steps
        self.proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1")

        self.model.to(device=device,dtype=torch.half)

        self.batch_size = batch_size
        self.n_samples = batch_size if n_samples is None else n_samples 
        self.device = device

        # Just get prompts from MSCOCO labels
        loader = create_loader('coco', self.n_samples, 64, deterministic=True)
        _, self.prompts = next(iter(loader))

    @torch.no_grad()
    def call_pickscore(self, prompts, images):
        inputs = self.proc(
            images = images,
            text=prompts,
            padding = 'max_length', truncation = True, max_length = 77, return_tensors='pt'
        ).to(device='cuda')

        img_emb = self.model.get_image_features(pixel_values = inputs.pixel_values.half())
        img_emb = F.normalize(img_emb, p = 2, dim = -1)

        text_emb = self.model.get_text_features(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask)
        text_emb = F.normalize(text_emb, p = 2, dim = -1)

        cos_sims = torch.einsum('bd,bd->b', img_emb, text_emb)
        return cos_sims.sum()

    @torch.no_grad()
    def __call__(self, sampler, model):
        old_steps = sampler.config.n_steps
        sampler.config.n_steps = self.n_sampling_steps

        pick_score_total = 0.

        total_batches = self.n_samples // self.batch_size

        print("Scoring...")
        for i in tqdm(range(total_batches)):
            prompt_batch = self.prompts[i*self.batch_size:(i+1)*self.batch_size]
            
            # Generate images and make them PIL
            images = sampler.sample(self.batch_size, model, prompt_batch) #[-1,1] [b,c,h,w]
            images = (images.clamp(-1,1)+1)/2
            images = (images * 255).byte().cpu().permute(0,2,3,1).numpy()
            pil_images = [Image.fromarray(img) for img in images]

            
            score = self.call_pickscore(prompt_batch, pil_images)
            pick_score_total += score.item()

            # Free up CUDA memory
            del images
            torch.cuda.empty_cache()
        
        sampler.config.n_steps = old_steps

        return pick_score_total / self.n_samples

def test_pickscore():
    class DummySampler:
        def sample(self, n_samples, model, prompts):
            noise = torch.randn(n_samples, 3, 512, 512)
            return torch.clamp(noise, -1, 1)

    # Create a dummy model (None in this case)
    model = None

    # Create the dummy sampler
    sampler = DummySampler()

    # Initialize PickScorer
    scorer = PickScorer(batch_size=4, n_samples=16)  # Adjust batch_size and n_samples as needed

    # Test PickScorer
    score = scorer(sampler, model)
    print(f"Total PickScore: {score}")

def test_fid():
    from dit.configs import SamplerConfig
    class DummySampler:
        def __init__(self):
            self.config = SamplerConfig()

        def sample(self, batch_size, model, prompts):
            return torch.rand(batch_size, 3, 224, 224).to('cuda')

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=10000):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.rand(3, 224, 224), f"Prompt {idx}"

    # Create dummy dataset and dataloader
    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Create a dummy model (None in this case)
    model = None

    # Create the dummy sampler
    sampler = DummySampler()

    # Initialize FIDScorer
    fid_scorer = FIDScorer(dataloader, cache_path='./test_fid.pkl', total_size=10000, batch_size=64)

    # Test FIDScorer
    fid_score = fid_scorer(sampler, model)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    test_fid()
        
