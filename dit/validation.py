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

class PickScorer:
    def __init__(self, batch_size : int = 256, n_samples : int = None, device = 'cuda'):
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

        scores = self.model.logit_scale.exp() * (text_emb @ img_emb.T)
        scores = scores.diag().sum().item()
        return scores

    @torch.no_grad()
    def __call__(self, sampler, model):
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
            pick_score_total += score

            # Free up CUDA memory
            del images
            torch.cuda.empty_cache()

        return pick_score_total / self.n_samples

if __name__ == "__main__":
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
        
        
