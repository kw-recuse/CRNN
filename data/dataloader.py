import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from datasets import load_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TextRecognitionDataset(Dataset):
    def __init__(self, hf_dataset, char_to_idx, target_height=32):
        self.dataset = hf_dataset
        self.char_to_idx = char_to_idx
        self.target_height = target_height
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((target_height, int(target_height * 100 / 32))), # resize height to 32 keeping aspect ratio
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # normalize to [-1, 1]
        ])
        
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        curr = self.dataset[idx]
        image = curr['image']
        label = curr['label']
        
        image = self.transform(image) # preprocess the images
        
        label_indices = [self.char_to_idx[char.lower()] for char in label if char.lower() in self.char_to_idx]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        label_length = torch.tensor(len(label_indices), dtype=torch.long)
        
        return image, label_tensor, label_length
    
    
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    
    max_width = max(img.size(2) for img in images)
    padded_images = torch.zeros(len(images), 1, 32, max_width)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.size(2)] = img
    
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.stack(label_lengths)
    
    return padded_images, labels, label_lengths


def create_dataloaders(batch_size):
    # download the dataset from HuggingFace
    dataset = load_dataset("priyank-m/MJSynth_text_recognition")
    
    hf_train_dataset = dataset["train"]
    hf_val_dataset = dataset["val"]
    hf_test_dataset = dataset["test"]
    
    CHARACTERS = [''] + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
    
    train_dataset = TextRecognitionDataset(hf_dataset=hf_train_dataset, char_to_idx=CHAR_TO_IDX, target_height=32)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    val_dataset = TextRecognitionDataset(hf_dataset=hf_val_dataset, char_to_idx=CHAR_TO_IDX, target_height=32)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    test_dataset = TextRecognitionDataset(hf_dataset=hf_test_dataset, char_to_idx=CHAR_TO_IDX, target_height=32)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    
    return train_dataloader, val_dataloader, test_dataloader