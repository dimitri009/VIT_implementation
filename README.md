# VIT_implementation
VIT implementation in PyTorch.
![vit](https://github.com/dimitri009/VIT_implementations/blob/main/img_VIT/VIT.png?raw=true)

[An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929) is a great paper and in this project we will try to explain and implement the architecture presented in the paper. 

### Install requirements 

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` as decribed [here](https://pytorch.org/get-started/locally/)

## Patches 
![patc](https://github.com/dimitri009/VIT_implementations/blob/main/img_VIT/Patches.png?raw=true)

The input tensor images is expected to have the shape:

`[batch_size, channels, height, width]`

where:

* `batch_size`: is the number of images in the batch.
* `channels`: is the number of color channels (e.g., 3 for RGB).
* `height`: and width are the spatial dimensions of the image.

The goal is to divide each image into smaller non-overlapping patches of size `patch_size x patch_size`, and prepare them for further processing.
To do that we will use three functions from PyTorch:

* `unfold` slices the image into patches.
* `permute` rearranges the dimensions for better organization.
* `reshape` flattens each patch into a single vector and groups all patches into a list.

          patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
          patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, -1, channels * self.patch_size * self.patch_size)

## PositionalEncoding
![pos](https://github.com/dimitri009/VIT_implementations/blob/main/img_VIT/Patches%2C%20Pos.png?raw=true)

Positional encoding is crucial in transformers because, unlike convolutional or recurrent architectures, transformers do not inherently process sequential or spatial relationships in data. Without this positional information, the model cannot distinguish between patches or their order. We will add to each "token" an information about their locations.

* `num_patches`:

    The total number of patches in the input image after dividing it into smaller non-overlapping patches (e.g., for a 32×32 image with 8×8 patches, there are 4×4=16 patches).
    Each patch is treated as a "token."

* `embedding_dim`:

    The size of the feature vector representing each patch. Each patch is embedded in a space of dimension `embedding_dim`.

* `positions`:

    A tensor containing integers from `0` to `num_patches - 1`. These integers represent the position of each patch in the image.

  Example:

    If num_patches = 16, positions = [0, 1, 2, ..., 15]

* `emb`:

    This is a learnable embedding layer (nn.Embedding) that maps each positional index (e.g., 0, 1, 2, ..., 15) into a vector of size embedding_dim.
    Think of it as a lookup table where:

  Input: Patch index (e.g., 0, 1, 2).

  Output: Learnable embedding vector (e.g., [0.2, -0.5, 1.0, ...] with length embedding_dim)

        positions = torch.arange(num_patches)
        emb = nn.Embedding(num_patches, embedding_dim)


## Transformer block
![tra](https://github.com/dimitri009/VIT_implementations/blob/main/img_VIT/Transformer.png?raw=true)

  The Transformer block is a core component of the Transformer architecture, which processes a sequence of tokens (or patches, in the case of Vision Transformers). Each block applies self-attention to understand the relationships between tokens and then refines these representations with a feed-forward neural network (MLP). This implementation also includes residual connections and layer normalization for better training stability.

 * `Layer Normalization`:
   ![3](https://github.com/dimitri009/VIT_implementations/blob/main/img_VIT/norm.png?raw=true)

        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-5)

Normalizes the input along the last dimension (`embedding_dim`) to stabilize training and improve gradient flow.
`eps=1e-5` prevents division by zero during normalization.

* `Multi-Head Self-Attention`:
  ![1](https://github.com/dimitri009/VIT_implementations/blob/main/img_VIT/m_attn.png?raw=true)

        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=2, dropout=drop_rate)

Computes self-attention, which allows the model to learn dependencies between tokens in the input sequence.
`num_heads=2` means the attention mechanism splits the input into two parallel attention heads.
`dropout=drop_rate` adds regularization by randomly dropping attention values.

* `Feed-Forward Network` (MLP):
  ![0](https://github.com/dimitri009/VIT_implementations/blob/main/img_VIT/mlp.png?raw=true)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(drop_rate)
        )

  A simple 2-layer MLP applied to each token independently:
  
  `Linear(embedding_dim, embedding_dim)`: Projects the input to the same dimensional space.
  `GELU()`: Applies a GELU activation function, which is smoother than ReLU.
  `Dropout(drop_rate)`: Adds regularization to prevent overfitting.
  Another `Linear and Dropout` complete the MLP.

# TEST 

We will make our experiments on the well known [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
![cifar10](https://github.com/dimitri009/VIT_implementation/blob/main/img_VIT/cifar10.png?raw=true)

As this is just a showcase we will limit our architecture:

* patch_size = 8
* embedding_dim = 256
* num_blocks = 2
  
Total Parameters: 847626

# RESULTS
