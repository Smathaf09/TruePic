"""
Vision Transformer (ViT) Model Architecture

This module implements a Vision Transformer for AI-generated image detection.
The ViT model excels at capturing global patterns and relationships in images
that might be missed by traditional CNNs.

Author: AI Image Detection Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class ViTClassifier(nn.Module):
    """
    Vision Transformer for AI-generated image detection.
    
    This model divides images into patches, processes them through transformer
    blocks, and makes classification decisions based on global attention patterns.
    """
    
    def __init__(self,
                 num_classes: int = 2,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.0):
        """
        Initialize the Vision Transformer.
        
        Args:
            num_classes: Number of output classes
            image_size: Input image size (assuming square images)
            patch_size: Size of each patch
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
            dropout_rate: Dropout rate
            attention_dropout: Attention dropout rate
        """
        super(ViTClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate,
                attention_dropout=attention_dropout
            ) for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"ViTClassifier initialized with {self._count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for different layer types."""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer norm and extract class token
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (B, embed_dim)
        
        # Classification
        output = self.head(cls_token_final)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention weights from a specific layer for visualization.
        
        Args:
            x: Input tensor
            layer_idx: Layer index to extract attention from (-1 for last layer)
            
        Returns:
            Attention weights tensor
        """
        batch_size = x.shape[0]
        
        # Forward pass until the specified layer
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks) + layer_idx:  # Handle negative indexing
                # Return attention weights from this block
                return block.get_attention_weights(x)
            x = block(x)
        
        # If we get here, return attention from the last block
        if self.blocks:
            return self.blocks[-1].get_attention_weights(x)
        else:
            return None
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get patch embeddings for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Patch embeddings
        """
        return self.patch_embed(x)
    
    def _count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            'model_name': 'ViTClassifier',
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'num_patches': self.num_patches,
            'total_parameters': self._count_parameters(),
            'architecture': 'Vision Transformer'
        }


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.
    
    This module divides the input image into non-overlapping patches
    and linearly projects them to the embedding dimension.
    """
    
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768):
        """
        Initialize patch embedding layer.
        
        Args:
            image_size: Size of input image
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Use convolution to extract patches and project to embedding space
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input image to patch embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        batch_size, channels, height, width = x.shape
        
        # Check input dimensions
        assert height == self.image_size and width == self.image_size, \
            f"Input image size ({height}x{width}) doesn't match expected ({self.image_size}x{self.image_size})"
        
        # Extract patches and project
        x = self.projection(x)  # (B, embed_dim, H//P, W//P)
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and MLP.
    
    This is the core building block of the Vision Transformer,
    implementing the standard transformer architecture.
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0):
        """
        Initialize transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=dim * mlp_ratio,
            dropout=dropout
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        x = x + self.dropout1(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization."""
        return self.attn.get_attention_weights(self.norm1(x))


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This module implements the core attention mechanism that allows
    the model to focus on different parts of the input simultaneously.
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of multi-head attention."""
        batch_size, num_tokens, dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, num_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization."""
        batch_size, num_tokens, dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        return attn  # (batch_size, num_heads, num_tokens, num_tokens)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron used in transformer blocks.
    
    This module implements the feed-forward network that processes
    the attention outputs in each transformer block.
    """
    
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.0):
        """
        Initialize MLP.
        
        Args:
            in_features: Input features dimension
            hidden_features: Hidden layer dimension
            out_features: Output features dimension
            activation: Activation function
            dropout: Dropout rate
        """
        super(MLP, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Factory function for creating different ViT variants
def create_vit_model(variant: str = 'base', **kwargs) -> ViTClassifier:
    """
    Factory function to create different ViT model variants.
    
    Args:
        variant: Model variant ('tiny', 'small', 'base', 'large')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized ViT model
    """
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'patch_size': 16,
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'patch_size': 16,
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'patch_size': 16,
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'patch_size': 16,
        }
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown ViT variant: {variant}. Available: {list(configs.keys())}")
    
    config = configs[variant]
    config.update(kwargs)
    
    return ViTClassifier(**config)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = ViTClassifier(num_classes=2)
    print(f"Model created with {model._count_parameters()} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test attention weights extraction
    attention_weights = model.get_attention_weights(dummy_input)
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test model info
    info = model.get_model_info()
    print("Model info:", info)