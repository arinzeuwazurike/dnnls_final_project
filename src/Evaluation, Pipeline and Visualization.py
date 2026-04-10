# Project Evaluation + BLEU
from nltk.translate.bleu_score import sentence_bleu
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project Evaluation + BLEU


def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    bleu_scores = []

    with torch.no_grad():
        for batch in dataloader:
            # --- Unpack batch ---
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

            # --- Forward pass ---
            _, _, predicted_text_logits_k, *_ = model(
                frames,
                input_ids_roberta,
                attention_mask_roberta,
                decoder_input_lstm
            )

            # --- Text loss ---
            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_flat = target_ids_lstm.reshape(-1)
            loss_text = criterion_text(prediction_flat, target_flat)
            total_loss += loss_text.item()

            # --- BLEU ---
            preds = torch.argmax(predicted_text_logits_k, dim=-1)
            for pred, tgt in zip(preds, target_ids_lstm):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                bleu = sentence_bleu([tgt_text.split()], pred_text.split())
                bleu_scores.append(bleu)

    return {
        "text_loss": total_loss / len(dataloader),
        "bleu": sum(bleu_scores) / len(bleu_scores)
    }


# Save predictions
def save_predictions(model, dataloader, num_samples=5):
    model.eval()
    samples = []

    with torch.no_grad():
        for batch in dataloader:
            frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch
            frames = frames.to(device)

            decoder_input_lstm = text_dict["decoder_input_ids"].to(device)
            target_ids_lstm = text_dict["target_ids"].to(device)

            _, _, predicted_text_logits_k, *_ = model(
                frames,
                text_dict["input_ids"].to(device),
                text_dict["attention_mask"].to(device),
                decoder_input_lstm
            )

            preds = torch.argmax(predicted_text_logits_k, dim=-1)

            for i in range(len(preds)):
                pred_text = tokenizer.decode(preds[i], skip_special_tokens=True)
                tgt_text = tokenizer.decode(target_ids_lstm[i], skip_special_tokens=True)

                samples.append({
                    "ground_truth": tgt_text,
                    "prediction": pred_text
                })

                if len(samples) >= num_samples:
                    return samples
    return samples

# Plot loss
def plot_loss_curve(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Roberta + CLIP Training Loss")
    plt.savefig("Roberta+CLIP_loss_curve.png")
    plt.close()


# BLEU bar chart
def plot_bleu(bleu_score):
    plt.figure()
    plt.bar(["Roberta+CLIP"], [bleu_score])
    plt.ylabel("BLEU Score")
    plt.title("Roberta + CLIP BLEU")
    plt.savefig("Roberta + CLIP_bleu.png")
    plt.close()


# Figure 3
def save_prediction_examples(model, dataloader, n_samples=5):
    model.eval()
    examples = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break

            # --- Correct unpack ---
            frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch

            frames = frames.to(device)
            image_target = image_target.to(device)

            input_ids = text_dict["input_ids"].to(device)
            attention_mask = text_dict["attention_mask"].to(device)
            decoder_input = text_dict["decoder_input_ids"].to(device)
            target_ids = text_dict["target_ids"].to(device)

            # --- Forward ---
            pred_image, _, pred_text_logits, *_ = model(
                frames,
                input_ids,
                attention_mask,
                decoder_input
            )

            pred_tokens = pred_text_logits.argmax(dim=-1)

            gt_text = tokenizer.decode(target_ids[0], skip_special_tokens=True)
            pred_text = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)

            examples.append({
                "input_frames": frames[0].cpu().tolist(),
                "ground_truth_image": image_target[0].cpu().tolist(),
                "predicted_image": pred_image[0].cpu().tolist(),
                "ground_truth_text": gt_text,
                "predicted_text": pred_text
            })

    return examples
# Figure 4
def plot_attention_heatmaps(model, dataloader, device, n_samples=3, save_dir="figures"):
    """
    - Shows n_samples individual attention maps (one per batch sample)
    - Shows an averaged attention map across batch for general trends
    - Saves all plots as PNGs
    """
    model.eval()
    sample_count = 0
    all_attn_maps = []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= n_samples:
                break

            # --- Correct unpack ---
            frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch

            frames = frames.to(device)
            input_ids = text_dict["input_ids"].to(device)
            attention_mask = text_dict["attention_mask"].to(device)
            decoder_input = text_dict["decoder_input_ids"].to(device)

            # --- Forward ---
            _, _, _, attn_weights, *_ = model(
                frames,
                input_ids,
                attention_mask,
                decoder_input
            )

            batch_size = attn_weights.size(0)

            # --- Individual maps ---
            for b in range(batch_size):
                if sample_count >= n_samples:
                    break

                attn_map = attn_weights[b].detach().cpu().numpy()

                title = f"Attention_Heatmap_Sample_{sample_count}"

                plt.figure(figsize=(6,5))
                plt.imshow(attn_map, cmap='viridis', aspect='auto')
                plt.title(title)
                plt.xlabel("Input tokens/frames")
                plt.ylabel("Output tokens/frames")
                plt.colorbar()

                save_path = os.path.join(save_dir, f"{title}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.show()

                plt.close()
                print(f"Saved: {save_path}")

                sample_count += 1

            all_attn_maps.append(attn_weights)

    # --- Average heatmap ---
    if all_attn_maps:
        all_attn_tensor = torch.cat(all_attn_maps, dim=0)
        avg_attn = all_attn_tensor.mean(dim=0)
        avg_attn_map = avg_attn.detach().cpu().numpy()

        title = f"Average_Attention_Map_{len(all_attn_tensor)}_Samples"

        plt.figure(figsize=(6,5))
        plt.imshow(avg_attn_map, cmap='viridis', aspect='auto')
        plt.title(title)
        plt.xlabel("Input tokens/frames")
        plt.ylabel("Output tokens/frames")
        plt.colorbar()

        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

        plt.close()
        print(f"Saved: {save_path}")