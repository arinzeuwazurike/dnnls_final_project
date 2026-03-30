# Project Evaluation + BLEU
from nltk.translate.bleu_score import sentence_bleu
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    bleu_scores = []

    with torch.no_grad():
        for (frames, descriptions, image_target, text_target,
             roi1, roi2, roi_valid, roi_frame, ent_id) in dataloader:

            frames = frames.to(device)
            descriptions = descriptions.to(device)
            text_target = text_target.to(device)

            _, _, predicted_text_logits_k, _, _, _, _ = model(frames, descriptions, text_target)

            # Loss
            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:]
            target_flat = target_labels.reshape(-1)

            loss = criterion_text(prediction_flat, target_flat)
            total_loss += loss.item()

            # BLEU
            preds = torch.argmax(predicted_text_logits_k, dim=-1)

            for pred, tgt in zip(preds, target_labels):
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
        for (frames, descriptions, image_target, text_target,
             roi1, roi2, roi_valid, roi_frame, ent_id) in dataloader:

            frames = frames.to(device)
            descriptions = descriptions.to(device)
            text_target = text_target.to(device)

            _, _, predicted_text_logits_k, _, _, _, _ = model(frames, descriptions, text_target)

            preds = torch.argmax(predicted_text_logits_k, dim=-1)

            for i in range(len(preds)):
                pred_text = tokenizer.decode(preds[i], skip_special_tokens=True)
                tgt_text = tokenizer.decode(text_target.squeeze(1)[i][1:], skip_special_tokens=True)

                samples.append({
                    "ground_truth": tgt_text,
                    "prediction": pred_text
                })

                if len(samples) >= num_samples:
                    return samples


# Plot loss
def plot_loss_curve(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline Training Loss")
    plt.savefig("baseline_loss_curve.png")
    plt.close()


# BLEU bar chart
def plot_bleu(bleu_score):
    plt.figure()
    plt.bar(["Baseline"], [bleu_score])
    plt.ylabel("BLEU Score")
    plt.title("Baseline BLEU")
    plt.savefig("baseline_bleu.png")
    plt.close()


# Figure 3
def save_prediction_examples(model, dataloader, n_samples=5):
    model.eval()
    examples = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break

            frames, descriptions, image_target, text_target, *_ = batch
            frames = frames.to(device)
            descriptions = descriptions.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)

            pred_image, _, pred_text_logits, *_ = model(frames, descriptions, text_target)
            pred_text_tokens = pred_text_logits.argmax(dim=-1)

            gt_text = tokenizer.decode(text_target.squeeze(1)[0][1:], skip_special_tokens=True)
            pred_text = tokenizer.decode(pred_text_tokens[0], skip_special_tokens=True)

            examples.append({
                "input_frames": frames[0].cpu().tolist(),
                "ground_truth_image": image_target[0].cpu().tolist(),
                "predicted_image": pred_image[0].cpu().tolist(),
                "ground_truth_text": gt_text,
                "predicted_text": pred_text
            })

    return examples


# Figure 4
def plot_attention_heatmaps(model, dataloader, device, n_samples=3):
    """
    - Shows n_samples individual attention maps (one per batch sample)
    - Shows an averaged attention map across batch for general trends
    """
    model.eval()
    sample_count = 0
    all_attn_maps = []

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= n_samples:
                break

            # Unpack batch
            frames, descriptions, image_target, text_target, *_ = batch
            frames, descriptions = frames.to(device), descriptions.to(device)

            # Forward pass
            _, _, _, attn_weights, *_ = model(frames, descriptions, text_target)
            # attn_weights shape: [batch, decoder_seq_len, encoder_seq_len]

            batch_size = attn_weights.size(0)

            # Plot individual attention maps ---
            for b in range(batch_size):
                if sample_count >= n_samples:
                    break
                attn_map = attn_weights[b].detach().cpu().numpy()  # [decoder_seq_len, encoder_seq_len]

                plt.figure(figsize=(6, 5))
                plt.imshow(attn_map, cmap='viridis', aspect='auto')
                plt.title(f"Attention Heatmap - Sample {sample_count}")
                plt.xlabel("Input tokens/frames")
                plt.ylabel("Output tokens/frames")
                plt.colorbar()
                plt.show()

                sample_count += 1

            # Collect for average heatmap
            all_attn_maps.append(attn_weights)

    # Plot averaged attention map
    if all_attn_maps:
        all_attn_tensor = torch.cat(all_attn_maps, dim=0)  # [total_samples, decoder_seq_len, encoder_seq_len]
        avg_attn = all_attn_tensor.mean(dim=0)  # average over all samples
        avg_attn_map = avg_attn.detach().cpu().numpy()

        plt.figure(figsize=(6, 5))
        plt.imshow(avg_attn_map, cmap='viridis', aspect='auto')
        plt.title(f"Average Attention Map over {len(all_attn_tensor)} Samples")
        plt.xlabel("Input tokens/frames")
        plt.ylabel("Output tokens/frames")
        plt.colorbar()
        plt.show()
print("Running final evaluation on test set...")

baseline_metrics = evaluate_model(sequence_predictor, test_dataloader)
print("Baseline Metrics:", baseline_metrics)

# Save metrics
with open("baseline_results.json", "w") as f:
    json.dump(baseline_metrics, f, indent=4)

# Save table
df = pd.DataFrame([{
    "Model": "Baseline",
    "Loss": baseline_metrics["text_loss"],
    "BLEU": baseline_metrics["bleu"]
}])
df.to_csv("baseline_table.csv", index=False)

# Plot metrics
plot_loss_curve(losses)
plot_bleu(baseline_metrics["bleu"])

# Save predictions
samples = save_predictions(sequence_predictor, test_dataloader)
with open("baseline_predictions.json", "w") as f:
    json.dump(samples, f, indent=4)

# Save detailed prediction examples for Figure 3
prediction_examples = save_prediction_examples(sequence_predictor, test_dataloader)
with open("baseline_prediction_examples.json", "w") as f:
    json.dump(prediction_examples, f, indent=4)

with open("baseline_prediction_examples.json") as f:
    examples = json.load(f)

print(examples[0])

def show_full_example(example):
    frames = np.array(example["input_frames"])
    gt_img = np.array(example["ground_truth_image"])
    pred_img = np.array(example["predicted_image"])

    fig, axes = plt.subplots(2, 4, figsize=(12,6))

    # Input frames
    for i in range(4):
        img = frames[i].transpose(1,2,0)
        axes[0,i].imshow(img)
        axes[0,i].axis('off')
        axes[0,i].set_title(f"Input {i+1}")

    # Ground truth image
    gt_img = gt_img.transpose(1,2,0)
    axes[1,1].imshow(gt_img)
    axes[1,1].set_title("Ground Truth Image")
    axes[1,1].axis('off')

    # Predicted image
    pred_img = pred_img.transpose(1,2,0)
    axes[1,2].imshow(pred_img)
    axes[1,2].set_title("Predicted Image")
    axes[1,2].axis('off')

    # Hide unused plots
    axes[1,0].axis('off')
    axes[1,3].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n--- TEXT ---")
    print("Ground Truth:\n", example["ground_truth_text"])
    print("\nPrediction:\n", example["predicted_text"])

for idx, example in enumerate(examples[:3]): # the first 3
    print(f"\n=== Example {idx+1} ===")
    show_full_example(example)