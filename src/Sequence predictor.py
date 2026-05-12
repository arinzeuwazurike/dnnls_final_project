# @title The main sequence predictor model
"""
This is the core architecture `SequencePredictor`.
1. **Encoders**: Uses the `image_encoder` and `text_encoder` to process the sequence of 4 input frames and descriptions.
2. **Temporal Fusion**: A GRU processes the sequence of fused (image+text) embeddings to capture temporal dynamics.
3. **Attention**: Applies attention over the sequence to summarize context.
4. **Decoders**: Predicts the *next* (5th) frame's image and text using `image_decoder` and `text_decoder`.
"""
#Modified
class SequencePredictor(nn.Module):
    def __init__(self, visual_autoencoder, text_autoencoder, latent_dim, gru_hidden_dim):
        super(SequencePredictor, self).__init__()

        # --- 1. Static Encoders ---
        self.image_encoder = visual_autoencoder.encoder
        self.text_encoder = text_autoencoder.encoder  # This is HuggingFace RobertaModel


        # --- 2. Temporal Encoder ---
        fusion_dim = latent_dim * 2  # z_visual + z_text
        self.temporal_rnn = nn.GRU(fusion_dim, latent_dim, batch_first=True)

        # --- 3. Attention ---
        self.attention = Attention(gru_hidden_dim)

        # --- 4. Final Projection ---
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, latent_dim),
            nn.ReLU()
        )

        # --- 5. Decoders ---
        self.image_decoder = visual_autoencoder.decoder
        self.text_decoder = text_autoencoder.decoder

        # For initializing decoder hidden/cell
        self.fused_to_h0 = nn.Linear(latent_dim, latent_dim)
        self.fused_to_c0 = nn.Linear(latent_dim, latent_dim)

    def forward(self, image_seq, input_ids_text_encoder, attention_mask_text_encoder, target_seq_text_decoder):
        # image_seq: [batch_size, K, C, H, W]
        batch_size, K, C, H, W = image_seq.shape
        # input_ids_text_encoder: [batch_size, K, max_len]
        # attention_mask_text_encoder: [batch_size, K, max_len]
        # target_seq_text_decoder: [batch_size, max_len - 1]
        if input_ids_text_encoder.dim() == 2:
          input_ids_text_encoder = input_ids_text_encoder.unsqueeze(1).repeat(1, K, 1)
          attention_mask_text_encoder = attention_mask_text_encoder.unsqueeze(1).repeat(1, K, 1)


        _, _, T_max = input_ids_text_encoder.shape


        # --- 1. Image Encoder ---
        img_flat = image_seq.view(batch_size * K, C, H, W)
        z_v_flat, low_feat, mid_feat, high_feat = self.image_encoder(img_flat)
        z_v_seq = z_v_flat.view(batch_size, K, -1) # [batch_size, K, latent_dim]


        # --- 2. Text Encoder ---
        # 2a. Encode text for EACH frame in the input sequence (for temporal fusion)
        input_ids_flat_for_roberta = input_ids_text_encoder.view(batch_size * K, T_max)
        attention_mask_flat_for_roberta = attention_mask_text_encoder.view(batch_size * K, T_max)

        roberta_latent_flat, _, _ = self.text_encoder(
            input_ids=input_ids_flat_for_roberta,
            attention_mask=attention_mask_flat_for_roberta
        )
        z_t_seq = roberta_latent_flat.view(batch_size, K, -1) # [batch_size, K, latent_dim]

        # 2b. Encode text for the FIRST frame for grounding losses (as per original comment's intention)
        input_ids_first_frame = input_ids_text_encoder[:, 0, :] # [batch_size, T_max]
        attention_mask_first_frame = attention_mask_text_encoder[:, 0, :] # [batch_size, T_max]

        z_t_grounding, _, _ = self.text_encoder(
            input_ids=input_ids_first_frame,
            attention_mask=attention_mask_first_frame
        ) # z_t_grounding is [batch_size, latent_dim]


        # --- 3. Fusion for temporal RNN ---
        z_fusion_seq = torch.cat((z_v_seq, z_t_seq), dim=2) # [batch_size, K, 2*latent_dim]

        zseq, h = self.temporal_rnn(z_fusion_seq) # zseq: [B, K, latent_dim], h: [1, B, latent_dim]
        h = h.squeeze(0) # h: [B, latent_dim]

        # --- 4. Attention ---
        context, attn_weights = self.attention(zseq) # context: [B, latent_dim], attn_weights: [B, K]

        # --- 5. Final latent for decoders ---
        z = self.projection(torch.cat((h, context), dim=1)) # z: [B, latent_dim]


        # --- 6. Image decoder ---
        # Reshape and aggregate low, mid, high features for the decoder
        low_feat_seq = low_feat.view(batch_size, K, 768, 14, 14)
        mid_feat_seq = mid_feat.view(batch_size, K, 768, 14, 14)
        high_feat_seq = high_feat.view(batch_size, K, 768, 14, 14)

        # Average across the sequence dimension (K) for decoder inputs
        low_feat_avg = low_feat_seq.mean(dim=1) # [B, 768, 14, 14]
        mid_feat_avg = mid_feat_seq.mean(dim=1) # [B, 768, 14, 14]
        high_feat_avg = high_feat_seq.mean(dim=1) # [B, 768, 14, 14]

        pred_image_content = self.image_decoder(z, low_feat_avg, mid_feat_avg, high_feat_avg)
        pred_image_context = pred_image_content # Assuming for simplicity or if image decoder only outputs content

        # --- 7. Text decoding (teacher forcing) ---
        h0 = self.fused_to_h0(z).unsqueeze(0).repeat(self.text_decoder.num_layers, 1, 1)
        c0 = self.fused_to_c0(z).unsqueeze(0).repeat(self.text_decoder.num_layers, 1, 1)
        predicted_text_logits_k, _hidden, _cell = self.text_decoder(target_seq_text_decoder, h0, c0)

        # Return h0, c0 for generation in validation
        # Include attn_weights in the return
        return pred_image_content, pred_image_context, predicted_text_logits_k, h0, c0, z_v_flat, z_t_grounding, attn_weights
