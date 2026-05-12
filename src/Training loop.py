# @title Training loop for the sequence predictor
"""
The main training loop:
1. Iterates over epochs and batches.
2. Performs the forward pass to get predictions and latent representations.
3. Computes the **Base Losses**: Image L1, Context MSE, Text CrossEntropy.
4. Computes **CoT Grounding Losses** (if data is valid):
   - `loss_reid`: Visual consistency for re-identified entities.
   - `loss_ground_mse`: Embedding alignment between ROI and text.
   - `loss_contrast`: Contrastive loss for ROI-Text alignment.
   - `loss_entity_pool`: Consistency within the batch for the same entity.
5. Backpropagates total loss and updates weights.
6. Runs the validation visualization at the end of each epoch.
"""

from torch.cuda.amp import autocast, GradScaler
# Instantiate the model, define loss and optimizer

# --- CoT-loss weights (added) ---
LAMBDA_REID = 0.10            # pulls same-entity ROIs together (student idea)
LAMBDA_GROUND_MSE = 0.10      # Option 2: frame-aware ROI↔text MSE grounding
LAMBDA_CONTRAST = 0.10        # Option 1: contrastive ROI↔text grounding (InfoNCE)
LAMBDA_ENTITY_POOL = 0.05     # Option 3: within-batch entity pooling loss
sequence_predictor.to(device)

N_EPOCHS = 5
# Freeze CLIP
for p in sequence_predictor.image_encoder.clip.parameters():
    p.requires_grad = False

for p in sequence_predictor.image_decoder.parameters():
    p.requires_grad = False

sequence_predictor.train()
losses = []

# -----------------------------
# AMP + Grad Accumulation
# -----------------------------

scaler = GradScaler()

ACCUM_STEPS = 4
optimizer.zero_grad()

checkpoint_dir = '/content/drive/MyDrive/DL_Checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filename = os.path.join(checkpoint_dir, 'roberta_clip_checkpoint.pth')

start_epoch = 0
if os.path.exists(checkpoint_filename):
    print(f"Loading checkpoint from {checkpoint_filename}")
    sequence_predictor, optimizer, loaded_epoch, _ = load_checkpoint_from_drive(
        sequence_predictor, optimizer,
        filename='roberta_clip_checkpoint.pth'
    )
    start_epoch = loaded_epoch + 1
    print(f"Resuming training from epoch {start_epoch}")

for epoch in range(N_EPOCHS):

    if epoch == 0 and start_epoch == 0:
        print("Unfreezing image decoder...")
        for p in sequence_predictor.image_decoder.parameters():
            p.requires_grad = True

    running_loss = 0.0

    for step, batch in enumerate(train_dataloader):

        frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch

        frames = frames.to(device)
        image_target = image_target.to(device)
        roi1 = roi1.to(device)
        roi2 = roi2.to(device)
        roi_valid = roi_valid.to(device)
        roi_frame = roi_frame.to(device)

        input_ids_roberta = text_dict["input_ids"].to(device)
        attention_mask_roberta = text_dict['attention_mask'].to(device)
        decoder_input_lstm = text_dict["decoder_input_ids"].to(device)
        target_ids_lstm = text_dict["target_ids"].to(device)

        # -----------------------------
        # AMP forward pass
        # -----------------------------
        with autocast():

            pred_image_content, pred_image_context, predicted_text_logits_k, \
            h0_dec, c0_dec, z_v_seq, z_t_seq, attn_weights = sequence_predictor(
                frames,
                input_ids_roberta,
                attention_mask_roberta,
                decoder_input_lstm
            )

            # Resize target safely
            image_target_resized = F.interpolate(
                image_target,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

            loss_im = criterion_images(pred_image_content, image_target_resized)

            # Resize input frames sequence to match pred_image_context resolution (224, 224)
            # frames shape: [B, K, C, H, W] -> flatten B*K to use interpolate
            B_fs, K_fs, C_fs, H_fs, W_fs = frames.shape
            frames_resized = F.interpolate(
                frames.view(B_fs * K_fs, C_fs, H_fs, W_fs),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).view(B_fs, K_fs, C_fs, 224, 224)

            mu_global = frames_resized.mean(dim=[0, 1])
            mu_global = mu_global.unsqueeze(0).expand_as(pred_image_context)
            loss_context = criterion_ctx(pred_image_context, mu_global)

            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_labels = target_ids_lstm.reshape(-1)
            loss_text = criterion_text(prediction_flat, target_labels)

            loss_reid = torch.tensor(0.0, device=device)
            loss_ground_mse = torch.tensor(0.0, device=device)
            loss_contrast = torch.tensor(0.0, device=device)
            loss_entity_pool = torch.tensor(0.0, device=device)

            if roi_valid.any():
                mask = roi_valid.bool()
                if mask.sum() > 0:
                    with torch.no_grad():
                        z_r1_global, _, _, _ = sequence_predictor.image_encoder(roi1[mask])
                        z_r2_global, _, _, _ = sequence_predictor.image_encoder(roi2[mask])

                    z_r1 = z_r1_global
                    z_r2 = z_r2_global

                    loss_reid = F.mse_loss(z_r1, z_r2)

                    if USE_FRAME_AWARE_GROUNDING:
                        z_t_match = z_t_seq[mask]
                        loss_ground_mse = F.mse_loss(z_r1, z_t_match)

                    if USE_CONTRASTIVE_ROI:
                        z_img = F.normalize(z_r1, dim=-1)
                        z_txt = F.normalize(z_t_match, dim=-1)
                        logits = (z_img @ z_txt.t()) / CONTRASTIVE_TAU
                        labels = torch.arange(logits.size(0), device=device)
                        loss_contrast = F.cross_entropy(logits, labels)

            W_IM = 4.0
            W_CTX = 1.0
            W_TXT = 0.7

            loss = (
                W_IM * loss_im +
                W_CTX * loss_context +
                W_TXT * loss_text +
                LAMBDA_REID * loss_reid +
                LAMBDA_GROUND_MSE * loss_ground_mse +
                LAMBDA_CONTRAST * loss_contrast +
                LAMBDA_ENTITY_POOL * loss_entity_pool
            )

            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()
        running_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0:
            from src.final_training import apply_gradient_clipping
            scaler.unscale_(optimizer)
            apply_gradient_clipping(sequence_predictor)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    epoch_loss = running_loss / len(train_dataloader)
    losses.append(epoch_loss)

    print(
        f"Epoch [{epoch+1}/{N_EPOCHS}] Loss: {epoch_loss:.4f} "
        f"(im={loss_im.item():.3f}, ctx={loss_context.item():.3f}, txt={loss_text.item():.3f}, "
        f"reid={float(loss_reid):.3f}, g_mse={float(loss_ground_mse):.3f}, "
        f"nce={float(loss_contrast):.3f}, entpool={float(loss_entity_pool):.3f})"
    )

    validation(sequence_predictor, val_dataloader)
    sequence_predictor.train()

    save_checkpoint_to_drive(
        sequence_predictor, optimizer,
        epoch, epoch_loss,
        filename='roberta_clip_checkpoint.pth'
    )