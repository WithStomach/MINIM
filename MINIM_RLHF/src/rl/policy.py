import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import logging
from typing import Dict, Tuple, List
import os
import numpy as np
from PIL import Image
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

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.embed_dim = config["policy_model"]["architecture"]["image_encoder"]["output_dim"]
        self.num_heads = config["policy_model"]["architecture"]["image_encoder"]["num_heads"]
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.attn_drop = nn.Dropout(config["policy_model"]["architecture"]["image_encoder"]["dropout"])
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(config["policy_model"]["architecture"]["image_encoder"]["dropout"])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        embed_dim = config["policy_model"]["architecture"]["image_encoder"]["output_dim"]
        mlp_dim = config["policy_model"]["architecture"]["image_encoder"]["mlp_dim"]
        dropout = config["policy_model"]["architecture"]["image_encoder"]["dropout"]
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        image_size = config["synthetic_system"]["image_size"][0]
        patch_size = config["policy_model"]["architecture"]["image_encoder"]["patch_size"]
        in_channels = 1
        embed_dim = config["policy_model"]["architecture"]["image_encoder"]["output_dim"]
        num_layers = config["policy_model"]["architecture"]["image_encoder"]["num_layers"]
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.transformer = nn.Sequential(*[
            TransformerBlock(config) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]

class FusionNetwork(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        hidden_dims = config["policy_model"]["architecture"]["fusion"]["hidden_dims"]
        dropout = config["policy_model"]["architecture"]["fusion"]["dropout"]
        
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class RLPolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = get_device(config)
        
        self.image_encoder = VisionTransformer(config)
        self.clip = CLIPModel.from_pretrained(config["model_paths"]["clip"])
        self.clip_processor = CLIPProcessor.from_pretrained(config["model_paths"]["clip"])
        
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.fusion = FusionNetwork(config)
        self.head = self._build_head(config)
        
        self.setup_training()
        self.to(self.device)
        
    def _build_head(self, config: Dict) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(config["policy_model"]["architecture"]["fusion"]["hidden_dims"][-1], 1),
            nn.Sigmoid()
        )
        
    def setup_training(self):
        trainable_params = list(self.image_encoder.parameters()) + \
                          list(self.fusion.parameters()) + \
                          list(self.head.parameters())
        
        stage = self.config["training"]["stage"]
        learning_rate = self.config["training"][f"stage{stage}"]["learning_rate"]
                          
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=self.config["policy_model"]["training"]["optimizer"]["weight_decay"]
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["training"]["max_epochs"],
            eta_min=1e-6
        )
        
    def forward(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        image_features = self.image_encoder(images)
        text_inputs = self.clip_processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        text_features = self.clip.get_text_features(**text_inputs)
        
        combined_features = torch.cat([image_features, text_features], dim=1)
        fused = self.fusion(combined_features)
        return self.head(fused)
        
    def compute_stage1_loss(self, scores: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        """π_new = π + α∑(R(s,a) - π(s,a))∇π(s,a)"""
        # Normalize ratings based on the custom rating range
        min_rating, max_rating = min(self.config["data"]["rating_range"]), max(self.config["data"]["rating_range"])
        ratings = (ratings - min_rating) / (max_rating - min_rating)
        
        advantage = (ratings - scores.squeeze()).abs()
        log_probs = torch.log(scores.squeeze() + 1e-8)
        return -torch.mean(advantage * log_probs)
        
    def compute_stage2_loss(self, scores: torch.Tensor, ratings: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """π_new = π + β∑(P_selected(xi)/P_π(a|s))(R(s,a) - V_π(s))∇π(s,a)"""
        # Normalize ratings based on the custom rating range
        min_rating, max_rating = min(self.config["data"]["rating_range"]), max(self.config["data"]["rating_range"])
        ratings = (ratings - min_rating) / (max_rating - min_rating)
        
        advantage = (ratings - value.squeeze()).abs()
        importance_weights = 1.0 
        log_probs = torch.log(scores.squeeze() + 1e-8)
        return -torch.mean(importance_weights * advantage * log_probs)
        
    def update(self, images: torch.Tensor, prompts: List[str], ratings: torch.Tensor, stage: int) -> float:
        self.train()
        self.optimizer.zero_grad()
        
        scores = self(images, prompts)
        value = self.compute_value(images, prompts) if stage == 2 else None
        loss = (
            self.compute_stage1_loss(scores, ratings) if stage == 1
            else self.compute_stage2_loss(scores, ratings, value)
        )
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def compute_value(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            return self(images, prompts)
        
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('L')
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:
            image = Image.fromarray(image).convert('L')
            image = np.array(image)
            
        image_pil = Image.fromarray(image)
        target_size = self.config["synthetic_system"]["image_size"]
        image_pil = image_pil.resize((target_size[0], target_size[1]))
        image = np.array(image_pil)
        
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        image_tensor = torch.tensor(image, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def evaluate_image(self, image, prompt: str) -> float:
        self.eval()
        with torch.no_grad():
            image_tensor = self.preprocess_image(image).to(self.device)
            score = self(image_tensor, [prompt]).item()
        return score
    