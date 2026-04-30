
# Project Evaluation
# Initialize metrics from HuggingFace evaluate
rouge_metric = evaluate.load('rouge')
meteor_metric = evaluate.load('meteor')

def evaluate_model(model, dataloader):
    model.eval()
    total_text_loss = 0
    total_image_loss = 0

    all_preds_text = []
    all_gts_text = []
    bleu_scores = []
    ssim_scores = []
    psnr_scores = []
    mse_scores = []

    with torch.no_grad():
        for batch in dataloader:
            # Adjusting unpacking to match requested structure
            # Note: The existing dataloader returns 9 items, mapping them accordingly
            frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch

            # Mapping to user names
            descriptions = text_dict["input_ids"].to(device)
            text_target = text_dict["target_ids"].to(device)
            frames = frames.to(device)
            image_target = image_target.to(device)

            # Forward pass: Capturing all 8 return values from SequencePredictor
            # but using the 3 arguments requested by user
            pred_image, _, predicted_text_logits_k, _, _, _, _, attn_weights = model(
                frames,
                descriptions,
                text_dict["attention_mask"].to(device),
                text_dict["decoder_input_ids"].to(device)
            )

            # Image Metrics
            img_loss = F.mse_loss(pred_image, image_target)
            total_image_loss += img_loss.item()

            for i in range(image_target.size(0)):
                gt_np = image_target[i].cpu().permute(1, 2, 0).numpy()
                pred_np = pred_image[i].cpu().permute(1, 2, 0).numpy()

                # SSIM
                s = ssim(gt_np, pred_np, data_range=1.0, channel_axis=2)
                ssim_scores.append(s)

                # MSE
                m = np.mean((gt_np - pred_np) ** 2)
                mse_scores.append(m)

                # PSNR
                p = psnr(gt_np, pred_np, data_range=1.0)
                psnr_scores.append(p)

            # Text Metrics
            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_flat = text_target.reshape(-1)
            loss_text = criterion_text(prediction_flat, target_flat)
            total_text_loss += loss_text.item()

            preds = torch.argmax(predicted_text_logits_k, dim=-1)
            for pred, tgt in zip(preds, text_target):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)

                all_preds_text.append(pred_text)
                all_gts_text.append(tgt_text)

                # BLEU
                bleu = sentence_bleu([tgt_text.split()], pred_text.split())
                bleu_scores.append(bleu)

    # Batch-level text metrics
    rouge_results = rouge_metric.compute(predictions=all_preds_text, references=all_gts_text)
    meteor_results = meteor_metric.compute(predictions=all_preds_text, references=all_gts_text)

    return {
        "text_loss": total_text_loss / len(dataloader),
        "image_reconstruction_loss": total_image_loss / len(dataloader),
        "bleu": sum(bleu_scores) / len(bleu_scores),
        "rougeL": rouge_results['rougeL'],
        "meteor": meteor_results['meteor'],
        "ssim": sum(ssim_scores) / len(ssim_scores),
        "mse": sum(mse_scores) / len(mse_scores),
        "psnr": sum(psnr_scores) / len(psnr_scores)
    }

def plot_attention_heatmaps(model, dataloader, device, n_samples=3):
    model.eval()
    all_attn_maps = []
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_idx >= n_samples: break
            frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch

            # Capturing attention weights (8th return value)
            _, _, _, _, _, _, _, attn_weights = model(
                frames.to(device),
                text_dict["input_ids"].to(device),
                text_dict["attention_mask"].to(device),
                text_dict["decoder_input_ids"].to(device)
            )

            for b in range(attn_weights.size(0)):
                if sample_idx >= n_samples: break
                # Reshape for the 4 input frames visualization
                attn_map = attn_weights[b].detach().cpu().numpy().reshape(1, -1)
                plt.figure(figsize=(6, 2))
                img = plt.imshow(attn_map, cmap='viridis', aspect='auto')
                plt.title(f"Attention Map Sample {sample_idx+1}\n(Importance over 4 input frames)")
                plt.xlabel("Input Frame Index (0-3)")
                plt.ylabel("Attention Value")
                cbar = plt.colorbar(img)
                cbar.set_label('Attention Weight')
                plt.show()
                all_attn_maps.append(attn_weights[b])
                sample_idx += 1

    if all_attn_maps:
        avg_attn = torch.stack(all_attn_maps).mean(dim=0).cpu().numpy().reshape(1, -1)
        plt.figure(figsize=(6, 2))
        img = plt.imshow(avg_attn, cmap='viridis', aspect='auto')
        plt.title(f"Average Attention Map ({n_samples} samples)\n(Mean importance over 4 input frames)")
        plt.xlabel("Input Frame Index (0-3)")
        plt.ylabel("Attention Value")
        cbar = plt.colorbar(img)
        cbar.set_label('Attention Weight')
        plt.show()

def get_prediction_examples(model, dataloader, n=5):
    model.eval()
    examples = []
    with torch.no_grad():
        for batch in dataloader:
            if len(examples) >= n: break
            frames, image_target, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict, obj_labels = batch

            pred_img, _, pred_logits, *_ = model(
                frames.to(device),
                text_dict["input_ids"].to(device),
                text_dict["attention_mask"].to(device),
                text_dict["decoder_input_ids"].to(device)
            )

            for i in range(frames.size(0)):
                if len(examples) >= n: break
                examples.append({
                    "input_frames": frames[i].cpu().tolist(),
                    "ground_truth_image": image_target[i].cpu().tolist(),
                    "predicted_image": pred_img[i].cpu().tolist(),
                    "ground_truth_text": tokenizer.decode(text_dict["target_ids"][i], skip_special_tokens=True),
                    "predicted_text": tokenizer.decode(pred_logits[i].argmax(dim=-1), skip_special_tokens=True)
                })
    return examples

def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Roberta + CLIP FIN 2 Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(metrics_dict):
    """
    Plots a bar chart for BLEU, ROUGE-L, METEOR, SSIM, and PSNR
    with value labels on top of each bar.
    """
    labels = ['BLEU', 'ROUGE-L', 'METEOR', 'SSIM', 'PSNR']
    values = [
        metrics_dict.get('bleu', 0),
        metrics_dict.get('rougeL', 0),
        metrics_dict.get('meteor', 0),
        metrics_dict.get('ssim', 0),
        metrics_dict.get('psnr', 0)
    ]

    # Ensure values are floats for plotting
    values = [float(v) if hasattr(v, 'item') else v for v in values]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['skyblue', 'salmon', 'lightgreen', 'orange', 'plum'])

    # Add exact figures on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}',
                 va='bottom', ha='center', fontweight='bold')

    plt.ylabel('Score')
    plt.title('Final Model Evaluation Metrics 2')
    # Set ylim slightly higher than max value to fit text
    plt.ylim(0, max(values) * 1.2 if values and max(values) > 0 else 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


print("Running final evaluation on test set...")

baseline_metrics = evaluate_model(sequence_predictor, test_dataloader)
print("Roberta + CLIP FIN 2 Metrics:", baseline_metrics)

# Convert NumPy types to standard Python floats for JSON serialization
serializable_metrics = {k: float(v) if hasattr(v, 'item') else v for k, v in baseline_metrics.items()}

# Save metrics
with open("Roberta+CLIP_FIN_2_results.json", "w") as f:
    json.dump(serializable_metrics, f, indent=4)

# Save table including new metrics
df = pd.DataFrame([{
    "Model": "Roberta+CLIP FIN 2",
    "Text_Loss": baseline_metrics["text_loss"],
    "Image_Loss": baseline_metrics["image_reconstruction_loss"],
    "BLEU": baseline_metrics["bleu"],
    "ROUGE-L": baseline_metrics["rougeL"],
    "METEOR": baseline_metrics["meteor"],
    "SSIM": baseline_metrics["ssim"],
    "PSNR": baseline_metrics["psnr"],
    "Number of Epochs": N_EPOCHS,
    "Learning Rate": lr,
    "Batch Size": batch_size,
    "Embedding Dim": emb_dim,
    "Latent Dim": latent_dim,
    "Num Layers": num_layers
}])

df.to_csv("Roberta+CLIP_FIN_2_table_and_parameters.csv", index=False)

# Plot curves
if 'losses' in locals():
    plot_loss_curve(losses)

# Use the multi-metric plot
plot_metrics(baseline_metrics)

# Save detailed prediction examples for Figure 3
prediction_examples = get_prediction_examples(sequence_predictor, test_dataloader, n=5)
with open("Roberta+CLIP_FIN_2_prediction_examples.json", "w") as f:
    json.dump(prediction_examples, f, indent=4)

print("Evaluation complete. Results saved.")

# Plot attention / explainability maps for Figure 4
plot_attention_heatmaps(sequence_predictor, test_dataloader, device)

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