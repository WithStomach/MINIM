import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from ..utils.medical_processor import MedicalImageProcessor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ..utils.device_utils import get_device, move_to_device

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn(num_patches + 1, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x

class Selector(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = get_device(config)
        
        image_size = config["synthetic_system"]["image_size"][0]
        patch_size = config["selector"]["architecture"]["patch_size"]
        in_channels = 1
        embed_dim = config["selector"]["architecture"]["embed_dim"]
        num_layers = config["selector"]["architecture"]["num_layers"]
        num_heads = config["selector"]["architecture"]["num_heads"]
        mlp_dim = config["selector"]["architecture"]["mlp_dim"]
        dropout = config["selector"]["architecture"]["dropout"]
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.transformer = self._build_transformer(embed_dim, num_heads, mlp_dim, dropout, num_layers)
        self.head = self._build_head(embed_dim, config)
        
        self = move_to_device(self, self.device)
        
    def _build_transformer(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, num_layers: int) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu'
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def _build_head(self, embed_dim: int, config: Dict) -> nn.Sequential:
        head_dims = [embed_dim] + config["selector"]["architecture"]["head"]["hidden_dims"]
        head_dropout = config["selector"]["architecture"]["head"]["dropout"]
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            *[item for i in range(len(head_dims)-1) for item in [
                nn.Linear(head_dims[i], head_dims[i+1]),
                nn.GELU(),
                nn.Dropout(head_dropout)
            ]],
            nn.Linear(head_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x[:, 0]
        score = self.head(x)
        return score

class ReceiveSelector:
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config["training"]["stage2"]["selector_threshold"]
        self.medical_processor = MedicalImageProcessor(config)
        self.model = Selector(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["selector"]["training"]["learning_rate"],
            weight_decay=config["selector"]["training"]["optimizer"]["weight_decay"]
        )
        
    def evaluate_image(self, image) -> float:
        processed_image = self.medical_processor.load_medical_image(image)
        metrics = self.medical_processor.compute_quality_metrics(processed_image)
        
        image_tensor = torch.tensor(processed_image).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.model.device)
        
        with torch.no_grad():
            score = self.model(image_tensor).item()
        
        if (metrics['snr'] >= self.config["selector"]["quality_metrics"]["snr_threshold"] and
            metrics['contrast'] >= self.config["selector"]["quality_metrics"]["contrast_threshold"]):
            return score
        return 0.0
    
    def filter_samples(self, images: List[str], ratings: List[float], prompts: List[str], threshold: Optional[float] = None) -> Tuple[List[str], List[float]]:
        selected_samples = []
        selected_ratings = []
        selected_prompts = []
        
        threshold = threshold or self.threshold
        
        for img_path, rating, prompt in zip(images, ratings, prompts):
            score = self.evaluate_image(img_path)
            if score >= threshold:
                selected_samples.append(img_path)
                selected_ratings.append(rating)
                selected_prompts.append(prompt)
                
        return selected_samples, selected_ratings, selected_prompts
    
    def train_step(self, image_paths: List[str], ratings: List[float], stage: int) -> torch.Tensor:
        images = []
        for path in image_paths:
            processed_image = self.medical_processor.load_medical_image(path)
            image_tensor = torch.tensor(processed_image).unsqueeze(0).unsqueeze(0).float()
            images.append(image_tensor)
        
        images = torch.cat(images, dim=0).to(self.model.device)
        ratings = torch.tensor(ratings, device=self.model.device)
        
        min_rating, max_rating = min(self.config["data"]["rating_range"]), max(self.config["data"]["rating_range"])
        ratings = (ratings - min_rating) / (max_rating - min_rating)
        
        scores = self.model(images)
        
        if stage == 1:
            advantage = (ratings - scores.squeeze()).abs()
            log_probs = torch.log(scores.squeeze() + 1e-8)
            loss = -torch.mean(advantage * log_probs)
        else:
            value = self.model(images).detach()
            advantage = (ratings - value.squeeze()).abs()
            importance_weights = 1.0
            log_probs = torch.log(scores.squeeze() + 1e-8)
            loss = -torch.mean(importance_weights * advantage * log_probs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss