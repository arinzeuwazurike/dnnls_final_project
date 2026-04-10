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

# Instantiate the model, define loss and optimizer

# --- CoT-loss weights (added) ---
LAMBDA_REID = 0.10            # pulls same-entity ROIs together (student idea)
LAMBDA_GROUND_MSE = 0.10      # Option 2: frame-aware ROI↔text MSE grounding
LAMBDA_CONTRAST = 0.10        # Option 1: contrastive ROI↔text grounding (InfoNCE)
LAMBDA_ENTITY_POOL = 0.05     # Option 3: within-batch entity pooling loss

# Modified
# BEFORE training loop Freeze CLIP
for p in sequence_predictor.image_encoder.clip.parameters():
    p.requires_grad = False

for p in sequence_predictor.image_decoder.parameters():
    p.requires_grad = False

sequence_predictor.train()
losses = []

for epoch in range(N_EPOCHS):
    if epoch == 2:
        print("Unfreezing image decoder...")
        for p in sequence_predictor.image_decoder.parameters():
            p.requires_grad = True
    running_loss = 0.0


    for batch in train_dataloader:
        # --- Unpack batch ---
        # Adjust indices to match your dataloader
        # Here, last item is text dict from TextTaskDataset
        frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch


        # --- Move tensors to device ---
        frames = frames.to(device)
        image_target = image_target.to(device)
        roi1 = roi1.to(device)
        roi2 = roi2.to(device)
        roi_valid = roi_valid.to(device)
        roi_frame = roi_frame.to(device)

        # Extract text inputs from text_dict
        input_ids_roberta = text_dict["input_ids"].to(device)
        attention_mask_roberta = text_dict['attention_mask'].to(device)
        decoder_input_lstm = text_dict["decoder_input_ids"].to(device)
        target_ids_lstm = text_dict["target_ids"].to(device)

        optimizer.zero_grad()

        # --- Forward pass ---
        pred_image_content, pred_image_context, predicted_text_logits_k, h0_dec, c0_dec, z_v_seq, z_t_seq, attn_weights = sequence_predictor(
            frames,
            input_ids_roberta,
            attention_mask_roberta,
            decoder_input_lstm
        )

        # -------------------------
        # Base losses
        # -------------------------
        loss_im = criterion_images(pred_image_content, image_target)

        mu_global = frames.mean(dim=[0, 1])
        mu_global = mu_global.unsqueeze(0).expand_as(pred_image_context)
        loss_context = criterion_ctx(pred_image_context, mu_global)

        prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
        target_labels = target_ids_lstm.reshape(-1) # Use target_ids_lstm for loss
        loss_text = criterion_text(prediction_flat, target_labels)

        # -------------------------
        # CoT-based grounding losses
        # -------------------------
        loss_reid = torch.tensor(0.0, device=device)
        loss_ground_mse = torch.tensor(0.0, device=device)
        loss_contrast = torch.tensor(0.0, device=device)
        loss_entity_pool = torch.tensor(0.0, device=device)

        if roi_valid.any():
            mask = roi_valid.bool()
            if mask.sum() > 0:
                z_r1 = sequence_predictor.image_encoder(roi1[mask])
                z_r2 = sequence_predictor.image_encoder(roi2[mask])

                # ReID grounding
                loss_reid = F.mse_loss(z_r1, z_r2)

                # Frame-aware grounding
                if USE_FRAME_AWARE_GROUNDING:
                    # z_t_seq shape is (B, latent_dim).
                    # Since we only encoded frame 0 text in the baseline architecture,
                    # we align the ROIs to that text embedding.
                    z_t_match = z_t_seq[mask]
                    loss_ground_mse = F.mse_loss(z_r1, z_t_match)


                # Contrastive ROI↔text grounding
                if USE_CONTRASTIVE_ROI:
                    z_img = F.normalize(z_r1, dim=-1)
                    z_txt = F.normalize(z_t_match, dim=-1) # z_t_match from above
                    logits = (z_img @ z_txt.t()) / CONTRASTIVE_TAU
                    labels = torch.arange(logits.size(0), device=device)
                    loss_contrast = F.cross_entropy(logits, labels)

                # Entity pooling
                if USE_ENTITY_POOLING:

                    if isinstance(ent_id, list) and all(isinstance(e, str) for e in ent_id): # Check if ent_id is actually a list of strings
                        ent_list = [ent_id[i] for i, m in enumerate(mask.detach().cpu().tolist()) if m]
                        uniq = {}
                        for i_e, eid in enumerate(ent_list):
                            if not eid: # Empty string IDs are ignored
                                continue
                            uniq.setdefault(eid, []).append(i_e)

                        if len(uniq) > 0:
                            pool_losses = []
                            for eid, idxs in uniq.items():
                                if len(idxs) < 2:
                                    continue
                                group = z_r1[idxs]
                                mean = group.mean(dim=0, keepdim=True)
                                pool_losses.append(F.mse_loss(group, mean.expand_as(group)))
                            if len(pool_losses) > 0:
                                loss_entity_pool = torch.stack(pool_losses).mean()


        # -------------------------
        # Total loss
        # -------------------------
        W_IM = 3.0
        W_CTX = 1.0
        W_TXT = 0.7 # to reduce text dominance
        loss = W_IM * loss_im + W_CTX * loss_context + W_TXT * loss_text
        loss = loss + LAMBDA_REID * loss_reid
        loss = loss + LAMBDA_GROUND_MSE * loss_ground_mse
        loss = loss + LAMBDA_CONTRAST * loss_contrast
        loss = loss + LAMBDA_ENTITY_POOL * loss_entity_pool


        # Backprop
        loss.backward()
        # Gradient clipping
        from src.final_training import apply_gradient_clipping
        apply_gradient_clipping(sequence_predictor)
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    losses.append(epoch_loss)

    print(
        f"Epoch [{epoch+1}/{N_EPOCHS}] Loss: {epoch_loss:.4f}  "
        f"(im={loss_im.item():.3f}, ctx={loss_context.item():.3f}, txt={loss_text.item():.3f}, "
        f"reid={float(loss_reid):.3f}, g_mse={float(loss_ground_mse):.3f}, "
        f"nce={float(loss_contrast):.3f}, entpool={float(loss_entity_pool):.3f})")

    # Validation step
    validation(sequence_predictor, val_dataloader)
    sequence_predictor.train()

    # Optional: Scheduler or Early stopping
    # if USE_SCHEDULER and scheduler is not None:
    #     scheduler.step(val_loss)
    # if USE_EARLY_STOPPING:
    #     stop = early_stopper.step(val_loss)
    #     if stop:
    #         break

