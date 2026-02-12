import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from decord import VideoReader, cpu
from PIL import Image

# --- Data Preparation ---
def prepare_dataframe(csv_path, metric_col='ECR', threshold=0.33):
    df = pd.read_csv(csv_path)
    
    # Normalize metric
    df[metric_col] = (df[metric_col] - df[metric_col].min()) / (df[metric_col].max() - df[metric_col].min())
    
    q1 = df[metric_col].quantile(threshold)
    q3 = df[metric_col].quantile(1 - threshold)
    
    def get_label(x):
        if x > q3: return 1
        elif x < q1: return 0
        else: return -1
        
    df['popularity'] = df[metric_col].apply(get_label)
    df = df[df['popularity'] != -1].reset_index(drop=True)
    return df

# --- TimeSformer Dataset ---
class TimesformerDataset(Dataset):
    def __init__(self, video_paths, labels, processor, num_frames=8):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            indices = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
            video_data = vr.get_batch(indices).asnumpy()
            inputs = self.processor(list(video_data), return_tensors="pt")
            return {"pixel_values": inputs['pixel_values'].squeeze(0), "label": label}
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return {"pixel_values": torch.zeros((self.num_frames, 3, 224, 224)), "label": label}

def collate_fn_timesformer(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# --- SmolVLM Helpers ---
def sample_frames_pil(video_path, num_frames=8):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except Exception:
        return [Image.new('RGB', (224, 224), color='black')] * num_frames

class SmolVLMDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self): return len(self.video_paths)
    def __getitem__(self, idx): return {"video": self.video_paths[idx], "label": self.labels[idx]}

def get_collate_fn_smolvlm(processor, label2id):
    def collate_fn(examples):
        instances = []
        labels = []
        MAX_FRAMES = 8
        
        for example in examples:
            pil_frames = sample_frames_pil(example["video"], num_frames=MAX_FRAMES)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": "Classify if this video is popular or not."},
                {"type": "video", "path": example["video"]} 
            ]}]
            text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            batch = processor(text=[text_inputs], videos=[[pil_frames]], padding=False, return_tensors="pt")
            
            instances.append({
                "input_ids": batch["input_ids"][0],
                "attention_mask": batch["attention_mask"][0],
                "pixel_values": batch["pixel_values"][0] if "pixel_values" in batch else None
            })
            
            lbl = example["label"]
            labels.append(label2id[lbl] if lbl in label2id else lbl)

        input_ids = pad_sequence([x["input_ids"] for x in instances], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        attention_mask = pad_sequence([x["attention_mask"] for x in instances], batch_first=True, padding_value=0)
        
        # Handle pixel values
        pvs = [x["pixel_values"] for x in instances if x["pixel_values"] is not None]
        if pvs:
            max_dim = max(p.shape[0] for p in pvs)
            c, h, w = pvs[0].shape[1:]
            padded_pvs = []
            for p in pvs:
                padded = torch.zeros((max_dim, c, h, w), dtype=p.dtype)
                padded[:p.shape[0]] = p
                padded_pvs.append(padded)
            pixel_values = torch.stack(padded_pvs)
        else:
            pixel_values = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    return collate_fn
