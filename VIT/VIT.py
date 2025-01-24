import torch
import torch.nn as nn
import torch.nn.functional as F



# Layer to extract patches from input image
class Patches(nn.Module):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, images):
        batch_size, channels, height, width = images.size()
        # Use unfold to extract patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1,
                                                            channels * self.patch_size * self.patch_size)
        return patches


# Positional Encoding for patches
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.positions = torch.arange(num_patches)
        self.emb = nn.Embedding(num_patches, embedding_dim)

    def forward(self, x):
        position_embedding = self.emb(self.positions.to(x.device))
        x = x + position_embedding
        return x


# Transformer block
class Transformer(nn.Module):
    def __init__(self, embedding_dim, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2, dropout=drop_rate)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        # Apply LayerNorm and attention
        x_norm = self.norm1(x)
        attention_output, _ = self.attention(x_norm, x_norm, x_norm)
        x1 = x + attention_output  # Residual connection

        # Apply MLP and LayerNorm
        x_norm2 = self.norm2(x1)
        mlp_output = self.mlp(x_norm2)
        return x1 + mlp_output  # Final residual connection


# Vision Transformer (ViT) Implementation
class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, embedding_dim=256, num_blocks=2, num_classes=10, drop_rate=0.1):
        super(ViT, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_dim = embedding_dim
        self.data_augmentation = nn.Sequential(
            nn.Identity()  # Augmentations can be added using torchvision transforms in the DataLoader.
        )
        self.encoder = nn.Sequential(
            Patches(patch_size, self.num_patches),
            nn.Linear(patch_size * patch_size * 3, embedding_dim),  # Project patches to embedding_dim
            PositionalEncoding(self.num_patches, embedding_dim)
        )
        self.transformers = nn.Sequential(
            *[Transformer(embedding_dim, drop_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = self.data_augmentation(x)
        encodings = self.encoder(x)
        encodings = encodings.permute(1, 0, 2)  # Change shape for MultiheadAttention (seq_len, batch_size, embedding_dim)
        features = self.transformers(encodings)
        logits = self.head(features.permute(1, 2, 0))  # Revert back for the classifier
        return logits