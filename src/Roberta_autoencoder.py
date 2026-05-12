import torch
import torch.nn as nn
import torch.nn.functional as F

# @title The text autoencoder (Seq2Seq)
#Modified
class RobertaEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=num_layers, dropout=dropout, unfreeze_layers=6):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # Freeze everything
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Unfreeze last N layers
        for layer in self.roberta.encoder.layer[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # =========================
        # Attention Pooling (nonlinear)
        # =========================
        self.attn_pool = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # =========================
        # Projection Head (512 → 256)
        # =========================
        self.projection = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),

            nn.Linear(256, hidden_dim)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden = outputs.last_hidden_state  # (B, T, 768)

        # =========================
        # Attention Pooling
        # =========================
        attn_scores = self.attn_pool(last_hidden)  # (B, T, 1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(attn_weights * last_hidden, dim=1)  # (B, 768)

        # =========================
        # CLS Token Fusion
        # =========================
        cls_token = last_hidden[:, 0]  # (B, 768)
        pooled = 0.7 * pooled + 0.3 * cls_token

        # =========================
        # Projection
        # =========================
        latent = self.projection(pooled)

        # Final normalization
        latent = self.layer_norm(latent)
        latent = self.dropout(latent)

        # =========================
        # LSTM-compatible output
        # =========================
        hidden = latent.unsqueeze(0).repeat(2, 1, 1) # Match 2-layer LSTM
        cell = torch.zeros_like(hidden)

        return latent, hidden, cell # Changed to return latent as the first output

#Modified to have two layers and weight tying
class DecoderLSTM(nn.Module):
    """
      Decodes a latent space representation into a sequence of tokens.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Added a small MLP head for better word prediction stability
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )

        self.out = nn.Linear(hidden_dim, vocab_size) # Should be hidden_dim

        # Weight Tying: Share weights between embedding and output linear layer
        self.out.weight = self.embedding.weight

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.head(output)
        prediction = self.out(output)
        return prediction, hidden, cell
#Modified
# We create the basic text autoencoder (a special case of a sequence to sequence model)
class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq, attention_mask=None):
        # Pass keyword args to RobertaEncoder
        _enc_out, hidden, cell = self.encoder(
            input_ids=input_seq,      # ← Keyword arg!
            attention_mask=attention_mask
        )

        # The target_seq argument (decoder_input_ids from dataset) is already input_ids[:-1].
        # No further slicing is needed here.
        decoder_input = target_seq
        predictions, _hidden, _cell = self.decoder(decoder_input, hidden, cell)
        return predictions