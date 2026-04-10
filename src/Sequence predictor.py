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
        self.text_decoder = text_autoencoder.decoder  # FIX: Correctly assign the LSTM text decoder

        # For initializing decoder hidden/cell
        self.fused_to_h0 = nn.Linear(latent_dim, latent_dim)
        self.fused_to_c0 = nn.Linear(latent_dim, latent_dim)

    def forward(self, image_seq, input_ids_text_encoder, attention_mask_text_encoder, target_seq_text_decoder):
        batch_size, seq_len, C, H, W = image_seq.shape

        # --- 1. Image Encoder ---
        img_flat = image_seq.view(batch_size * seq_len, C, H, W)
        z_v_flat = self.image_encoder(img_flat)  # [b*s, latent_dim]
        z_v_seq = z_v_flat.view(batch_size, seq_len, -1)  # [b, s, latent]

        # --- 2. Text Encoder with attention mask and roberta model---
        # FIX: Call self.text_encoder directly and use the appropriate input_ids/attention_mask
        # The RobertaEncoder's forward returns (None, hidden, cell)
        _, hidden_roberta, cell_roberta = self.text_encoder(
            input_ids=input_ids_text_encoder, # Using the argument for RoBERTa input
            attention_mask=attention_mask_text_encoder
        )
        # z_t_seq now represents the latent of the *first* frame's description text for each batch item
        z_t_seq = hidden_roberta.squeeze(0)   # (batch, latent_dim)
        z_t_flat = z_t_seq.unsqueeze(1).repeat(1, seq_len, 1).view(-1, z_t_seq.size(-1))

        # --- 3. Fusion for temporal RNN ---
        z_fusion_flat = torch.cat((z_v_flat, z_t_flat), dim=1)
        z_fusion_seq = z_fusion_flat.view(batch_size, seq_len, -1)

        zseq, h = self.temporal_rnn(z_fusion_seq)
        h = h.squeeze(0)

        # --- 4. Attention ---
        context, attn_weights = self.attention(zseq) # Capture attention weights here

        # --- 5. Final latent for decoders ---
        z = self.projection(torch.cat((h, context), dim=1))

        # --- 6. Image decoding ---
        pred_image_content, pred_image_context = self.image_decoder(z)

        # --- 7. Text decoding (teacher forcing) ---
        h0 = self.fused_to_h0(z).unsqueeze(0)
        c0 = self.fused_to_c0(z).unsqueeze(0)

        #  target
        decoder_input_for_lstm = target_seq_text_decoder # Direct use of the target sequence

        predicted_text_logits_k, _hidden, _cell = self.text_decoder(decoder_input_for_lstm, h0, c0)

        # Return h0, c0 for generation in validation
        # Include attn_weights in the return
        return pred_image_content, pred_image_context, predicted_text_logits_k, h0, c0, z_v_seq, z_t_seq, attn_weights