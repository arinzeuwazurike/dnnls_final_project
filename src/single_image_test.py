# @title Single Image Overfit Test
"""
This tests whether the model has enough capacity
to perfectly reconstruct one image.

If the model cannot overfit one image,
it will not reconstruct a full dataset properly.
"""

visual_autoencoder.train()

# =========================================================
# Get ONE image
# =========================================================

single_img = next(iter(autoencoder_dataloader))[0][0:1].to(device)

# Resize to CLIP resolution
single_img = F.interpolate(
    single_img,
    size=(224, 224),
    mode='bilinear',
    align_corners=False
)

# =========================================================
# Optimizer
# =========================================================

optimizer_test = torch.optim.AdamW(
    visual_autoencoder.parameters(),
    lr=1e-4
)

# =========================================================
# Loss
# =========================================================

criterion = ReconstructionLoss(
    pixel_weight=0.8,
    perceptual_weight=0.2
).to(device)

# =========================================================
# Training
# =========================================================

N_STEPS = 1500

loss_history = []

for step in range(N_STEPS):

    optimizer_test.zero_grad()

    # Forward pass
    reconstructed = visual_autoencoder(single_img)

    # Reconstruction loss
    loss = criterion(reconstructed, single_img)

    # Backward
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        visual_autoencoder.parameters(),
        1.0
    )

    optimizer_test.step()

    loss_history.append(loss.item())

    # =====================================================
    # Logging
    # =====================================================

    if step % 50 == 0:

        print(
            f"Step {step}/{N_STEPS} "
            f"| Loss: {loss.item():.4f}"
        )

    # =====================================================
    # Visualization
    # =====================================================

    if step % 200 == 0:

        visual_autoencoder.eval()

        with torch.no_grad():

            reconstructed = visual_autoencoder(single_img)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # Original
            ax[0].imshow(
                single_img[0]
                .detach()
                .cpu()
                .permute(1,2,0)
                .clamp(0,1)
            )

            ax[0].set_title("Original")
            ax[0].axis("off")

            # Reconstruction
            ax[1].imshow(
                reconstructed[0]
                .detach()
                .cpu()
                .permute(1,2,0)
                .clamp(0,1)
            )

            ax[1].set_title(
                f"Step {step}"
            )

            ax[1].axis("off")

            plt.show()

        visual_autoencoder.train()

# =========================================================
# Final Loss Curve
# =========================================================

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Single Image Overfit Test")
plt.show()