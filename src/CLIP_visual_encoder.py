# @title The visual autoencoder
"""
Defines the computer vision modules:
1. `Backbone`: A CNN that processes input images into feature maps.
2. `VisualEncoder`: Uses two backbones to separate 'content' and 'context' features, projecting them to a latent space.
3. `VisualDecoder`: Reconstructs images from the latent representation using Transposed Convolutions.
4. `VisualAutoencoder`: The container class for the encoder and decoder.
"""

# =========================================================
# Residual Block
# =========================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# =========================================================
# CLIP Encoder
# =========================================================

class CLIPEncoderWrapper(nn.Module):

    def __init__(self, latent_dim, unfreeze_layers=2):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )

        self.clip.vision_model.config.output_hidden_states = True
        self.clip.config.output_hidden_states = True

        # Freeze CLIP
        for p in self.clip.parameters():
            p.requires_grad = False

        # Unfreeze last layers
        if unfreeze_layers > 0:
            for layer in self.clip.vision_model.encoder.layers[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        hidden_dim = self.clip.config.vision_config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):

        # Resize to CLIP input
        x = F.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )

        mean = torch.tensor(
            [0.481, 0.457, 0.408],
            device=x.device
        ).view(1,3,1,1)

        std = torch.tensor(
            [0.268, 0.261, 0.275],
            device=x.device
        ).view(1,3,1,1)

        x = (x - mean) / std


        vision_outputs = self.clip.vision_model(
            pixel_values=x,
            return_dict=True
        )


        # Global latent
        z = self.projection(
            vision_outputs.pooler_output
        )

        # Spatial features
        low_feat = (
            vision_outputs.hidden_states[3][:, 1:, :]
            .transpose(1, 2)
            .reshape(-1, 768, 14, 14)
        )

        mid_feat = (
            vision_outputs.hidden_states[6][:, 1:, :]
            .transpose(1, 2)
            .reshape(-1, 768, 14, 14)
        )

        high_feat = (
            vision_outputs.hidden_states[9][:, 1:, :]
            .transpose(1, 2)
            .reshape(-1, 768, 14, 14)
        )

        return z, low_feat, mid_feat, high_feat

# =========================================================
# Decoder
# =========================================================

class VisualDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        # =====================================================
        # Global latent projection
        # =====================================================

        self.fc = nn.Linear(latent_dim, 256 * 14 * 14)

        # =====================================================
        # Skip projections
        # =====================================================

        self.low_skip  = nn.Conv2d(768, 64, kernel_size=1)
        self.mid_skip  = nn.Conv2d(768, 64, kernel_size=1)
        self.high_skip = nn.Conv2d(768, 64, kernel_size=1)

        # =====================================================
        # Fusion layer
        # =====================================================

        self.initial_fusion = nn.Conv2d(
            256 + 64 + 64 + 64,
            256,
            kernel_size=3,
            padding=1
        )

        # =====================================================
        # Decoder blocks
        # =====================================================

        self.up1 = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResidualBlock(128)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualBlock(64)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualBlock(32)
        )

        # STAGE (112 → 224)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.GroupNorm(8, 16),
            nn.GELU(),
            ResidualBlock(16)
        )

        # Final RGB output
        self.final = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, low_feat, mid_feat, high_feat):

        B = z.size(0)

        # Global latent → feature map
        x = self.fc(z).view(B, 256, 14, 14)

        # Skip features
        low  = self.low_skip(low_feat)
        mid  = self.mid_skip(mid_feat)
        high = self.high_skip(high_feat)

        # Fusion
        x = torch.cat([x, low, mid, high], dim=1)
        x = self.initial_fusion(x)

        # Progressive upsampling
        x = self.up1(x)   # 14 → 28
        x = self.up2(x)   # 28 → 56
        x = self.up3(x)   # 56 → 112
        x = self.up4(x)   # 112 → 224

        x = self.final(x)

        return x


# =========================================================
# Autoencoder
# =========================================================

class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = CLIPEncoderWrapper(latent_dim)
        self.decoder = VisualDecoder(latent_dim)

    def forward(self, x):

        z, low, mid, high = self.encoder(x)

        x_hat = self.decoder(z, low, mid, high)

        return x_hat

# =========================================================
# Perceptual Loss
# =========================================================

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)

        self.features = nn.Sequential(
            *list(vgg.features[:16])
        ).eval()

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, pred, target):

        pred_feat = self.features(pred)
        target_feat = self.features(target)

        return F.l1_loss(pred_feat, target_feat)

# =========================================================
# Combined Reconstruction Loss
# =========================================================

class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        pixel_weight=0.7,
        perceptual_weight=0.3
    ):
        super().__init__()

        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight

        self.perceptual = PerceptualLoss()

    def forward(self, pred, target):

        pixel_loss = (
            0.5 * F.mse_loss(pred, target)
            +
            0.5 * F.l1_loss(pred, target)
        )

        perceptual_loss = self.perceptual(pred, target)

        total = (
            self.pixel_weight * pixel_loss
            +
            self.perceptual_weight * perceptual_loss
        )

        return total
