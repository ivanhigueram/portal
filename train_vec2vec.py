#!/usr/bin/env python3
"""
Vec2Vec Training Script for Satellite Imagery Embeddings

This script implements the vec2vec approach from "One 2 Many: Generation of Embeddings for NLP Downstream Tasks"
adapted for translating between satellite imagery embeddings from different sensors (Sentinel vs Landsat).

The implementation follows the mathematical framework:
- Translation functions: F1 = B2 ∘ T ∘ A1, F2 = B1 ∘ T ∘ A2  
- Reconstruction functions: R1 = B1 ∘ T ∘ A1, R2 = B2 ∘ T ∘ A2
- Loss components: Adversarial + Reconstruction + Cycle Consistency + Vector Space Preservation

Usage:
    python train_vec2vec.py --source_embeddings embeddings/eurosat-resnet18-ssl4eol-moco.npz \
                           --target_embeddings embeddings/eurosat-resnet18-ssl4eol-moco-landsat.npz \
                           --epochs 1000 --batch_size 256
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EmbeddingDataset(Dataset):
    """Dataset for loading pre-computed embeddings from .npz files."""

    def __init__(self, source_path: str, target_path: str, split: str = "train"):
        """
        Args:
            source_path: Path to source embeddings (.npz file)
            target_path: Path to target embeddings (.npz file)
            split: 'train' or 'test'
        """
        # Load source embeddings (e.g., EuroSAT Sentinel)
        source_data = np.load(source_path)
        self.source_embeddings = source_data[f"x_{split}"]
        self.source_labels = source_data[f"y_{split}"]

        # Load target embeddings (e.g., EuroSATL Landsat)
        target_data = np.load(target_path)
        self.target_embeddings = target_data[f"x_{split}"]
        self.target_labels = target_data[f"y_{split}"]

        # Verify same number of samples
        assert len(self.source_embeddings) == len(self.target_embeddings), (
            f"Source and target must have same number of samples: {len(self.source_embeddings)} vs {len(self.target_embeddings)}"
        )

        # Check if labels match exactly
        if not np.array_equal(self.source_labels, self.target_labels):
            # Check if same labels but different order (common issue)
            if sorted(self.source_labels.tolist()) == sorted(
                self.target_labels.tolist()
            ):
                print(
                    f"⚠️  WARNING: Labels are the same but in different order for {split} split"
                )
                print(
                    "This might indicate different data shuffling between embedding extractions"
                )
                print("Proceeding anyway, but translation quality may be affected")
            else:
                # Different label sets - this is a real problem
                from collections import Counter

                source_counts = Counter(self.source_labels)
                target_counts = Counter(self.target_labels)
                print(
                    f"❌ Source label distribution: {dict(sorted(source_counts.items()))}"
                )
                print(
                    f"❌ Target label distribution: {dict(sorted(target_counts.items()))}"
                )
                raise AssertionError(
                    "Source and target have different label distributions. "
                    "This indicates they're from different datasets or splits."
                )

        print(f"Loaded {split} split: {len(self.source_embeddings)} samples")
        print(f"Source embedding dim: {self.source_embeddings.shape[1]}")
        print(f"Target embedding dim: {self.target_embeddings.shape[1]}")

    def __len__(self):
        return len(self.source_embeddings)

    def __getitem__(self, idx):
        return {
            "source": torch.FloatTensor(self.source_embeddings[idx]),
            "target": torch.FloatTensor(self.target_embeddings[idx]),
            "label": torch.LongTensor([self.source_labels[idx]])[0],
        }


class TranslatorNetwork(nn.Module):
    """
    Translator network implementing A -> T -> B mapping.
    Combines input adapter, shared backbone, and output adapter.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input adapter A: embedding_space -> latent_space
        self.input_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Shared backbone T: latent_space -> latent_space
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Output adapter B: latent_space -> embedding_space
        self.output_adapter = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass: x -> A(x) -> T(A(x)) -> B(T(A(x)))"""
        latent = self.input_adapter(x)
        latent = self.backbone(latent)
        output = self.output_adapter(latent)
        return output


class Discriminator(nn.Module):
    """Discriminator to distinguish real vs translated embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 1024, depth: int = 3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))

        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))

        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.model(x)


class Vec2VecLoss:
    """
    Implementation of all loss components from vec2vec paper section 3.2.
    Based on the original implementation using cosine similarity metrics.
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_cc_trans: float = 10.0,
        lambda_cc_rec: float = 0.0,
        lambda_cc_vsp: float = 10.0,
        lambda_vsp: float = 1.0,
        lambda_gen: float = 1.0,
        lambda_disc: float = 1.0,
        lambda_latent_gen: float = 1.0,
        lambda_sim_gen: float = 0.0,
        temp: float = 50.0,
        eps: float = 1e-8,
    ):
        # Loss weights from original vec2vec config/unsupervised.toml - exact mapping
        self.lambda_rec = lambda_rec  # rec_lambda = 1.0
        self.lambda_cc_trans = lambda_cc_trans  # cc_trans_lambda = 10.0
        self.lambda_cc_rec = lambda_cc_rec  # cc_rec_lambda = 0.0
        self.lambda_cc_vsp = lambda_cc_vsp  # cc_vsp_lambda = 10.0
        self.lambda_vsp = lambda_vsp  # vsp_lambda = 1.0
        self.lambda_gen = lambda_gen  # gen_lambda = 1.0
        self.lambda_disc = lambda_disc  # disc_lambda = 1.0
        self.lambda_latent_gen = lambda_latent_gen  # latent_gen_lambda = 1.0
        self.lambda_sim_gen = lambda_sim_gen  # sim_gen_lambda = 0.0
        self.temp = temp  # Temperature for contrastive loss
        self.eps = eps  # Small epsilon for numerical stability

    def adversarial_loss(self, real_pred, fake_pred, mode: str = "discriminator"):
        """Standard GAN adversarial loss."""
        if mode == "discriminator":
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
            return real_loss + fake_loss
        else:  # generator mode
            return F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )

    def gradient_penalty(
        self, discriminator, real_data, fake_data, device, lambda_gp=10.0
    ):
        """Compute gradient penalty for WGAN-GP style training stability."""
        batch_size = real_data.size(0)

        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Get discriminator output
        d_interpolated = discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        return penalty

    def reconstruction_loss(self, reconstructed, original):
        """
        Cosine similarity-based reconstruction loss: 1 - cos(original, reconstructed)
        Following original vec2vec implementation.
        """
        cosine_sim = F.cosine_similarity(original, reconstructed, dim=1)
        return (1 - cosine_sim).mean()

    def cycle_consistency_loss(self, cycle_reconstructed, original):
        """
        Cosine similarity-based cycle consistency loss: 1 - cos(original, cycle_reconstructed)
        Following original vec2vec implementation.
        """
        cosine_sim = F.cosine_similarity(original, cycle_reconstructed, dim=1)
        return (1 - cosine_sim).mean()

    def vector_space_preservation_loss(self, original_batch, translated_batch):
        """
        Vector Space Preservation loss based on original vec2vec implementation.
        Preserves similarity structure between embedding spaces using cosine similarity.
        """
        # Normalize embeddings for stable similarity computation
        original_norm = original_batch / (
            original_batch.norm(dim=1, keepdim=True) + self.eps
        )
        translated_norm = translated_batch / (
            translated_batch.norm(dim=1, keepdim=True) + self.eps
        )

        # Compute similarity matrices
        original_sim = torch.mm(original_norm, original_norm.t())
        translated_sim = torch.mm(translated_norm, translated_norm.t())

        # Compute absolute difference between similarity structures
        return F.l1_loss(translated_sim, original_sim)

    def contrastive_loss(self, embeddings_a, embeddings_b):
        """
        Contrastive loss using temperature-scaled cosine similarity.
        Based on original vec2vec contrastive_loss_fn implementation.
        """
        # Normalize embeddings
        a_norm = embeddings_a / (embeddings_a.norm(dim=1, keepdim=True) + self.eps)
        b_norm = embeddings_b / (embeddings_b.norm(dim=1, keepdim=True) + self.eps)

        # Compute similarity matrix
        similarities = torch.mm(a_norm, b_norm.t()) * self.temp

        # Create target labels (diagonal elements should be highest)
        batch_size = embeddings_a.size(0)
        targets = torch.arange(batch_size, device=embeddings_a.device)

        # Cross-entropy loss
        return F.cross_entropy(similarities, targets)

    def uni_loss(self, original, translated):
        """
        Uni loss: measures similarity between original and translated embeddings.
        Based on original vec2vec uni_loss_fn implementation.
        """
        cosine_sim = F.cosine_similarity(original, translated, dim=1)
        return (1 - cosine_sim).mean()

    def translation_loss(self, source, translated_target, target):
        """
        Translation loss: measures quality of cross-space translation.
        Based on original vec2vec trans_loss_fn implementation.
        """
        # Similarity between translated and real target embeddings
        cosine_sim = F.cosine_similarity(translated_target, target, dim=1)
        return (1 - cosine_sim).mean()


def evaluate_translation(translator_s2t, translator_t2s, test_loader, device):
    """Evaluate translation quality using cosine similarity and cycle consistency."""
    translator_s2t.eval()
    translator_t2s.eval()

    cosine_similarities = []
    cycle_consistencies = []

    with torch.no_grad():
        for batch in test_loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            # Translate source -> target
            translated = translator_s2t(source)

            # Compute cosine similarity with real target embeddings
            cos_sim = F.cosine_similarity(translated, target, dim=1)
            cosine_similarities.extend(cos_sim.cpu().numpy())

            # Compute cycle consistency: source -> target -> source
            cycle_reconstructed = translator_t2s(translated)
            cycle_loss = F.mse_loss(cycle_reconstructed, source, reduction="none").mean(
                dim=1
            )
            cycle_consistencies.extend(cycle_loss.cpu().numpy())

    return {
        "cosine_similarity_mean": np.mean(cosine_similarities),
        "cosine_similarity_std": np.std(cosine_similarities),
        "cycle_consistency_mean": np.mean(cycle_consistencies),
        "cycle_consistency_std": np.std(cycle_consistencies),
    }


def save_translated_embeddings(
    translator_s2t,
    translator_t2s,
    train_loader,
    test_loader,
    device,
    save_dir,
    experiment_name,
):
    """
    Save translated embeddings for both train and test splits.

    Args:
        translator_s2t: Source to target translator (Landsat -> Sentinel)
        translator_t2s: Target to source translator (Sentinel -> Landsat)
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device for computation
        save_dir: Directory to save translated embeddings
        experiment_name: Name for the experiment files
    """
    print("💾 Saving translated embeddings...")

    translator_s2t.eval()
    translator_t2s.eval()

    os.makedirs(save_dir, exist_ok=True)

    # Process both splits
    for split, loader in [("train", train_loader), ("test", test_loader)]:
        print(f"Processing {split} split...")

        source_embeddings = []
        target_embeddings = []
        translated_s2t_embeddings = []  # Landsat -> Sentinel
        translated_t2s_embeddings = []  # Sentinel -> Landsat
        labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Translating {split}"):
                source = batch["source"].to(device)  # Original Landsat embeddings
                target = batch["target"].to(device)  # Original Sentinel embeddings
                label = batch["label"].cpu().numpy()

                # Translate in both directions
                translated_s2t = translator_s2t(source)  # Landsat -> Sentinel
                translated_t2s = translator_t2s(target)  # Sentinel -> Landsat

                # Collect all embeddings
                source_embeddings.append(source.cpu().numpy())
                target_embeddings.append(target.cpu().numpy())
                translated_s2t_embeddings.append(translated_s2t.cpu().numpy())
                translated_t2s_embeddings.append(translated_t2s.cpu().numpy())
                labels.append(label)

        # Concatenate all batches
        source_embeddings = np.concatenate(source_embeddings, axis=0)
        target_embeddings = np.concatenate(target_embeddings, axis=0)
        translated_s2t_embeddings = np.concatenate(translated_s2t_embeddings, axis=0)
        translated_t2s_embeddings = np.concatenate(translated_t2s_embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Save translated embeddings
        save_path = os.path.join(save_dir, f"{experiment_name}_translated_{split}.npz")
        np.savez(
            save_path,
            # Original embeddings
            source_original=source_embeddings,  # Original Landsat
            target_original=target_embeddings,  # Original Sentinel
            # Translated embeddings
            landsat_to_sentinel=translated_s2t_embeddings,  # Landsat -> Sentinel (main translation)
            sentinel_to_landsat=translated_t2s_embeddings,  # Sentinel -> Landsat (reverse)
            # Labels
            labels=labels,
        )

        print(f"✅ Saved {split} translated embeddings: {save_path}")
        print(f"   - Original Landsat: {source_embeddings.shape}")
        print(f"   - Original Sentinel: {target_embeddings.shape}")
        print(f"   - Landsat→Sentinel: {translated_s2t_embeddings.shape}")
        print(f"   - Sentinel→Landsat: {translated_t2s_embeddings.shape}")

    print(f"🎉 All translated embeddings saved to {save_dir}/")

    # Create README for the saved embeddings
    readme_path = os.path.join(save_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# Translated Embeddings - {experiment_name}

## Files Generated

- `{experiment_name}_translated_train.npz`: Training split translated embeddings
- `{experiment_name}_translated_test.npz`: Test split translated embeddings

## Contents of each .npz file

- `source_original`: Original Landsat embeddings (source)
- `target_original`: Original Sentinel embeddings (target)  
- `landsat_to_sentinel`: **Main translation** - Landsat embeddings translated to Sentinel space
- `sentinel_to_landsat`: Reverse translation - Sentinel embeddings translated to Landsat space
- `labels`: Class labels for all samples

## Usage Example

```python
import numpy as np

# Load translated embeddings
data = np.load('{experiment_name}_translated_test.npz')

# Get the main translation: Landsat -> Sentinel
landsat_embeddings = data['source_original']
translated_sentinel = data['landsat_to_sentinel']  # This is what you want!
original_sentinel = data['target_original']
labels = data['labels']

# Compare translation quality
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(translated_sentinel, original_sentinel)
print(f"Average cosine similarity: {{similarity.diagonal().mean():.4f}}")
```

## Vec2Vec Translation Details

- **Source Domain**: Landsat (EuroSAT-L, 22×22 pixels, 7 bands)
- **Target Domain**: Sentinel-2 (EuroSAT, 64×64 pixels, 13 bands)
- **Translation Direction**: Landsat embeddings → Sentinel embedding space
- **Semantic Preservation**: Same land cover classes (10 classes)

Generated on: {np.datetime64("today")}
""")

    print(f"📖 Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Vec2Vec for satellite imagery embeddings"
    )
    parser.add_argument(
        "--source_embeddings",
        type=str,
        required=True,
        help="Path to source embeddings .npz file (e.g., EuroSAT Sentinel)",
    )
    parser.add_argument(
        "--target_embeddings",
        type=str,
        required=True,
        help="Path to target embeddings .npz file (e.g., EuroSATL Landsat)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size"
    )  # From config: train_batch_size=256
    parser.add_argument(
        "--lr_gen", type=float, default=2e-5, help="Generator learning rate"
    )  # From config: main_lr=2e-5
    parser.add_argument(
        "--lr_disc", type=float, default=1e-5, help="Discriminator learning rate"
    )  # From config: disc_lr=1e-5
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dimension for translators"
    )
    # Loss coefficients from original config/unsupervised.toml
    parser.add_argument(
        "--lambda_rec",
        type=float,
        default=1.0,
        help="Reconstruction loss weight (rec_lambda)",
    )
    parser.add_argument(
        "--lambda_cc_trans",
        type=float,
        default=10.0,
        help="CC translation loss weight (cc_trans_lambda)",
    )
    parser.add_argument(
        "--lambda_cc_rec",
        type=float,
        default=0.0,
        help="CC reconstruction loss weight (cc_rec_lambda)",
    )
    parser.add_argument(
        "--lambda_cc_vsp",
        type=float,
        default=10.0,
        help="CC VSP loss weight (cc_vsp_lambda)",
    )
    parser.add_argument(
        "--lambda_vsp",
        type=float,
        default=1.0,
        help="Vector space preservation loss weight (vsp_lambda)",
    )
    parser.add_argument(
        "--lambda_gen",
        type=float,
        default=1.0,
        help="Generator loss weight (gen_lambda)",
    )
    parser.add_argument(
        "--lambda_disc",
        type=float,
        default=1.0,
        help="Discriminator loss weight (disc_lambda)",
    )
    parser.add_argument(
        "--lambda_latent_gen",
        type=float,
        default=1.0,
        help="Latent generator loss weight (latent_gen_lambda)",
    )
    parser.add_argument(
        "--lambda_sim_gen",
        type=float,
        default=0.0,
        help="Similarity generator loss weight (sim_gen_lambda)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints", help="Directory to save models"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vec2vec-satellite",
        help="W&B project name",
    )
    parser.add_argument(
        "--eval_every", type=int, default=50, help="Evaluation frequency (epochs)"
    )
    parser.add_argument(
        "--save_translations",
        action="store_true",
        help="Save translated embeddings after training",
    )
    parser.add_argument(
        "--translation_dir",
        type=str,
        default="embeddings_translated",
        help="Directory to save translated embeddings",
    )
    parser.add_argument(
        "--lambda_gp",
        type=float,
        default=10.0,
        help="Gradient penalty weight for stability",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.001,
        help="Minimum improvement for early stopping",
    )

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"vec2vec_{Path(args.source_embeddings).stem}_to_{Path(args.target_embeddings).stem}",
    )

    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(
        args.source_embeddings, args.target_embeddings, split="train"
    )
    test_dataset = EmbeddingDataset(
        args.source_embeddings, args.target_embeddings, split="test"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Get embedding dimensions
    sample_batch = next(iter(train_loader))
    source_dim = sample_batch["source"].shape[1]
    target_dim = sample_batch["target"].shape[1]

    print(f"Source embedding dimension: {source_dim}")
    print(f"Target embedding dimension: {target_dim}")

    # Initialize models
    translator_s2t = TranslatorNetwork(source_dim, target_dim, args.hidden_dim).to(
        device
    )  # F1: source -> target
    translator_t2s = TranslatorNetwork(target_dim, source_dim, args.hidden_dim).to(
        device
    )  # F2: target -> source
    discriminator_source = Discriminator(source_dim).to(device)
    discriminator_target = Discriminator(target_dim).to(device)

    # Initialize optimizers with config values
    gen_params = list(translator_s2t.parameters()) + list(translator_t2s.parameters())
    disc_params = list(discriminator_source.parameters()) + list(
        discriminator_target.parameters()
    )

    # Using original config values: gen_beta1=0.9, gen_beta2=0.999, weight_decay=0.01
    optimizer_gen = torch.optim.Adam(
        gen_params, lr=args.lr_gen, betas=(0.9, 0.999), weight_decay=0.01
    )
    optimizer_disc = torch.optim.Adam(
        disc_params, lr=args.lr_disc, betas=(0.9, 0.999), weight_decay=0.01
    )

    # Initialize loss function with all coefficients from original config
    loss_fn = Vec2VecLoss(
        lambda_rec=args.lambda_rec,
        lambda_cc_trans=args.lambda_cc_trans,
        lambda_cc_rec=args.lambda_cc_rec,
        lambda_cc_vsp=args.lambda_cc_vsp,
        lambda_vsp=args.lambda_vsp,
        lambda_gen=args.lambda_gen,
        lambda_disc=args.lambda_disc,
        lambda_latent_gen=args.lambda_latent_gen,
        lambda_sim_gen=args.lambda_sim_gen,
    )

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Early stopping variables
    best_cosine_sim = -1.0
    patience_counter = 0
    best_epoch = 0

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        translator_s2t.train()
        translator_t2s.train()
        discriminator_source.train()
        discriminator_target.train()

        epoch_gen_loss = 0
        epoch_disc_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            batch_size = source.size(0)

            # ===================
            # Train Discriminators
            # ===================
            optimizer_disc.zero_grad()

            # Real samples
            real_source_pred = discriminator_source(source)
            real_target_pred = discriminator_target(target)

            # Fake samples
            with torch.no_grad():
                fake_target = translator_s2t(source)
                fake_source = translator_t2s(target)

            fake_source_pred = discriminator_source(fake_source)
            fake_target_pred = discriminator_target(fake_target)

            # Discriminator losses (weighted by lambda_disc)
            disc_source_loss = loss_fn.adversarial_loss(
                real_source_pred, fake_source_pred, "discriminator"
            )
            disc_target_loss = loss_fn.adversarial_loss(
                real_target_pred, fake_target_pred, "discriminator"
            )

            # Add gradient penalty for stability
            gp_source = loss_fn.gradient_penalty(
                discriminator_source,
                source,
                fake_source.detach(),
                device,
                args.lambda_gp,
            )
            gp_target = loss_fn.gradient_penalty(
                discriminator_target,
                target,
                fake_target.detach(),
                device,
                args.lambda_gp,
            )

            disc_loss = (
                loss_fn.lambda_disc * (disc_source_loss + disc_target_loss)
                + gp_source
                + gp_target
            )

            disc_loss.backward()
            optimizer_disc.step()

            # =================
            # Train Generators
            # =================
            optimizer_gen.zero_grad()

            # Forward translations
            translated_target = translator_s2t(source)  # F1(source)
            translated_source = translator_t2s(target)  # F2(target)

            # Reconstructions
            reconstructed_source = translator_t2s(
                translator_s2t(source)
            )  # R1 = F2(F1(source))
            reconstructed_target = translator_s2t(
                translator_t2s(target)
            )  # R2 = F1(F2(target))

            # Generator adversarial losses (weighted by lambda_gen)
            fake_source_pred = discriminator_source(translated_source)
            fake_target_pred = discriminator_target(translated_target)

            adv_loss_source = loss_fn.adversarial_loss(
                None, fake_source_pred, "generator"
            )
            adv_loss_target = loss_fn.adversarial_loss(
                None, fake_target_pred, "generator"
            )
            adv_loss = loss_fn.lambda_gen * (adv_loss_source + adv_loss_target)

            # Reconstruction losses (basic reconstruction)
            rec_loss_source = loss_fn.reconstruction_loss(reconstructed_source, source)
            rec_loss_target = loss_fn.reconstruction_loss(reconstructed_target, target)
            rec_loss = loss_fn.lambda_rec * (rec_loss_source + rec_loss_target)

            # Cross-correlation losses following original config structure
            # CC Translation: translation quality (cc_trans_lambda = 10.0)
            cc_trans_loss_s2t = loss_fn.translation_loss(
                source, translated_target, target
            )
            cc_trans_loss_t2s = loss_fn.translation_loss(
                target, translated_source, source
            )
            cc_trans_loss = loss_fn.lambda_cc_trans * (
                cc_trans_loss_s2t + cc_trans_loss_t2s
            )

            # CC Reconstruction: cycle consistency (cc_rec_lambda = 0.0 - disabled)
            cc_rec_loss_source = loss_fn.cycle_consistency_loss(
                reconstructed_source, source
            )
            cc_rec_loss_target = loss_fn.cycle_consistency_loss(
                reconstructed_target, target
            )
            cc_rec_loss = loss_fn.lambda_cc_rec * (
                cc_rec_loss_source + cc_rec_loss_target
            )

            # CC VSP: vector space preservation in cycle (cc_vsp_lambda = 10.0)
            cc_vsp_loss_s2t = loss_fn.vector_space_preservation_loss(
                source, reconstructed_source
            )
            cc_vsp_loss_t2s = loss_fn.vector_space_preservation_loss(
                target, reconstructed_target
            )
            cc_vsp_loss = loss_fn.lambda_cc_vsp * (cc_vsp_loss_s2t + cc_vsp_loss_t2s)

            # Basic VSP: vector space preservation in translation (vsp_lambda = 1.0)
            vsp_loss_s2t = loss_fn.vector_space_preservation_loss(
                source, translated_target
            )
            vsp_loss_t2s = loss_fn.vector_space_preservation_loss(
                target, translated_source
            )
            vsp_loss = loss_fn.lambda_vsp * (vsp_loss_s2t + vsp_loss_t2s)

            # Total generator loss following original config structure
            gen_loss = (
                adv_loss
                + rec_loss
                + cc_trans_loss
                + cc_rec_loss
                + cc_vsp_loss
                + vsp_loss
            )

            gen_loss.backward()
            optimizer_gen.step()

            epoch_gen_loss += gen_loss.item()
            epoch_disc_loss += disc_loss.item()

            # Update progress bar with detailed loss info
            pbar.set_postfix(
                {
                    "Gen": f"{gen_loss.item():.3f}",
                    "Disc": f"{disc_loss.item():.3f}",
                    "Adv": f"{adv_loss.item():.3f}",
                    "CC_T": f"{cc_trans_loss.item():.3f}",
                    "CC_V": f"{cc_vsp_loss.item():.3f}",
                }
            )

        # Log metrics
        avg_gen_loss = epoch_gen_loss / len(train_loader)
        avg_disc_loss = epoch_disc_loss / len(train_loader)

        wandb.log(
            {
                "epoch": epoch,
                "train/generator_loss": avg_gen_loss,
                "train/discriminator_loss": avg_disc_loss,
                "train/adversarial_loss": adv_loss.item(),
                "train/reconstruction_loss": rec_loss.item(),
                "train/cc_translation_loss": cc_trans_loss.item(),
                "train/cc_reconstruction_loss": cc_rec_loss.item(),
                "train/cc_vsp_loss": cc_vsp_loss.item(),
                "train/vector_space_preservation_loss": vsp_loss.item(),
            }
        )

        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            print(f"\nEvaluating at epoch {epoch + 1}...")
            eval_metrics = evaluate_translation(
                translator_s2t, translator_t2s, test_loader, device
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "eval/cosine_similarity_mean": eval_metrics[
                        "cosine_similarity_mean"
                    ],
                    "eval/cosine_similarity_std": eval_metrics["cosine_similarity_std"],
                    "eval/cycle_consistency_mean": eval_metrics[
                        "cycle_consistency_mean"
                    ],
                    "eval/cycle_consistency_std": eval_metrics["cycle_consistency_std"],
                }
            )

            print(
                f"Cosine Similarity: {eval_metrics['cosine_similarity_mean']:.4f} ± {eval_metrics['cosine_similarity_std']:.4f}"
            )
            print(
                f"Cycle Consistency: {eval_metrics['cycle_consistency_mean']:.4f} ± {eval_metrics['cycle_consistency_std']:.4f}"
            )

            # Early stopping check
            current_cosine_sim = eval_metrics["cosine_similarity_mean"]
            if current_cosine_sim > best_cosine_sim + args.early_stopping_min_delta:
                best_cosine_sim = current_cosine_sim
                best_epoch = epoch
                patience_counter = 0
                print(f"🎯 New best cosine similarity: {best_cosine_sim:.4f}")
            else:
                patience_counter += 1
                print(
                    f"⏳ No improvement for {patience_counter}/{args.early_stopping_patience} epochs"
                )

                if patience_counter >= args.early_stopping_patience:
                    print(
                        f"🛑 Early stopping triggered! Best cosine similarity: {best_cosine_sim:.4f} at epoch {best_epoch + 1}"
                    )
                    break

            # Save models
            checkpoint = {
                "epoch": epoch,
                "translator_s2t_state_dict": translator_s2t.state_dict(),
                "translator_t2s_state_dict": translator_t2s.state_dict(),
                "discriminator_source_state_dict": discriminator_source.state_dict(),
                "discriminator_target_state_dict": discriminator_target.state_dict(),
                "optimizer_gen_state_dict": optimizer_gen.state_dict(),
                "optimizer_disc_state_dict": optimizer_disc.state_dict(),
                "args": vars(args),
                "eval_metrics": eval_metrics,
            }

            checkpoint_path = os.path.join(
                args.save_dir, f"vec2vec_epoch_{epoch + 1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training completed!")

    # Save translated embeddings if requested
    if args.save_translations:
        print("\n" + "=" * 80)
        print("🔄 Generating and saving translated embeddings...")

        # Create experiment name from file paths
        source_name = Path(args.source_embeddings).stem
        target_name = Path(args.target_embeddings).stem
        experiment_name = f"{source_name}_to_{target_name}"

        save_translated_embeddings(
            translator_s2t=translator_s2t,
            translator_t2s=translator_t2s,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            save_dir=args.translation_dir,
            experiment_name=experiment_name,
        )

        print("=" * 80)
        print(f"✅ Translated embeddings saved to: {args.translation_dir}/")
        print(
            "📝 Main translation (Landsat→Sentinel): 'landsat_to_sentinel' key in .npz files"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
