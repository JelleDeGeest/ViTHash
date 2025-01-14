# ViTHash/hashing.py

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ViTHasher:
    """
    A simple class that loads openai/clip-vit-large-patch14 from Hugging Face
    and uses it to create an image hash (embedding), and compute distances.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def compute_hash(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]

    def compute_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        # 1 - cosine similarity of normalized vectors
        cosine_sim = np.dot(hash1, hash2)
        return 1.0 - cosine_sim
    
    def compute_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        cosine_sim = np.dot(hash1, hash2)
        return cosine_sim
    