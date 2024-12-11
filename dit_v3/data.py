import os
import json
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import random
from pathlib import Path

class FrameDataset(IterableDataset):
    """
    Dataset of embedded frames + controls from gameplay
    """
    def __init__(self, data_dir, image_size=256, frame_count=100):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.frame_count = frame_count
        
        # Find all unique data object IDs by looking at _vt.pt files
        vt_files = list(self.data_dir.glob("*_video.pt"))
        self.data_ids = [f.stem.replace('_video','') for f in vt_files]
        
        # Verify all required files exist
        for id in self.data_ids:
            print(self.data_dir / f"{id}_video.pt")
            assert (self.data_dir / f"{id}_video.pt").exists(), f"Missing video file for {id}"
            assert (self.data_dir / f"{id}_controls.pt").exists(), f"Missing controls file for {id}"
            assert (self.data_dir / f"{id}_info.json").exists(), f"Missing meta file for {id}"

        # Load metadata and sort data_ids by src_vid_id and src_vid_pos
        id_to_meta = {}
        for id in self.data_ids:
            meta_path = self.data_dir / f"{id}_info.json"
            with open(meta_path) as f:
                meta = json.load(f)
                id_to_meta[id] = meta

        # Sort data_ids based on src_vid_id first, then src_vid_pos
        self.data_ids.sort(key=lambda x: (
            id_to_meta[x]["src_vid_id"],
            id_to_meta[x]["src_vid_pos"]
        ))

        # Load video lengths from meta files
        self.vid_lens = {}
        for id in self.data_ids:
            meta_path = self.data_dir / f"{id}_info.json"
            with open(meta_path) as f:
                meta = json.load(f)
                self.vid_lens[id] = meta["vid_len"]
        
        # Create a mapping of cumulative frame counts up to each data_id
        self.frames_upto = {}
        cumulative_frames = 0
        for id in self.data_ids:
            self.frames_upto[id] = cumulative_frames
            cumulative_frames += self.vid_lens[id]

        self.total_samples = 0
        for id in self.vid_lens:
            self.total_samples += (self.vid_lens[id] - self.frame_count + 1)

    def __iter__(self):
        """
        Yields file paths, start and end idx
        """
        while True:
            # Get random data object ID
            data_id = random.choice(self.data_ids)
            
            # Get sequence length from meta
            seq_len = self.vid_lens[data_id]
            
            if seq_len < self.frame_count:
                continue
                
            # Calculate slice indices
            start_idx = random.randint(0, seq_len - self.frame_count)
            end_idx = start_idx + self.frame_count
            
            # Return paths and indices
            vt_path = self.data_dir / f"{data_id}_video.pt"
            it_path = self.data_dir / f"{data_id}_controls.pt"
            
            yield {
                'vt_path': vt_path,
                'it_path': it_path,
                'start_idx': start_idx,
                'end_idx': end_idx
            }

    def create_loader(self, batch_size, num_workers=4):
        def collate_fn(batch):
            """
            Smart loading to make sure we don't load same file twice
            """
            # Control files are assumed to be small enough that it doesn't matter
            vt_paths = [x['vt_path'] for x in batch]
            it_paths = [x['it_path'] for x in batch]
            start_inds = [x['start_idx'] for x in batch]
            end_inds = [x['end_idx'] for x in batch]

            res_vid = []
            res_ctrl = []

            # Each file, which idx's is it serving?
            file_map = {}
            for i, path in enumerate(vt_paths):
                if path in file_map:
                    file_map[path].append(i)
                else:
                    file_map[path] = [i]
            
            for path in file_map:
                vid = torch.load(path)
                ctrl = torch.load(str(path).replace("_video.pt", "_controls.pt"))

                for i in file_map[path]:
                    slice_start = start_inds[i]
                    slice_end = end_inds[i]

                    res_vid.append(vid[slice_start:slice_end])
                    res_ctrl.append(ctrl[slice_start:slice_end])
        
            res_vid = torch.stack(res_vid)
            res_ctrl = torch.stack(res_ctrl)

            return res_vid, res_ctrl

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )


if __name__ == "__main__":
    dataset = FrameDataset("./train_data")
    dataloader = dataset.create_loader(batch_size=4, num_workers=0)
    
    # Get first batch
    vt_batch, it_batch = next(iter(dataloader))
    
    print(f"Visual tokens batch shape: {vt_batch.shape}")
    print(f"Image tokens batch shape: {it_batch.shape}")
