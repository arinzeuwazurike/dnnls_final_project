# Modified
# Plots four images and their reconstructions
def validation(model, data_loader):
    model.eval()
    with torch.no_grad():
        # Unpack 9 values from the DataLoader output
        frames, image_target_val, roi1, roi2, roi_valid, roi_frame, ent_id, text_dict_val, obj_labels = next(
            iter(data_loader))

        # Move tensors to device
        frames = frames.to(device)
        image_target_val = image_target_val.to(device)

        # Unpack text inputs for the model
        input_ids_roberta_val = text_dict_val["input_ids"].to(device)
        attention_mask_roberta_val = text_dict_val["attention_mask"].to(device)
        decoder_input_lstm_val = text_dict_val["decoder_input_ids"].to(device)
        target_ids_val = text_dict_val["target_ids"].to(device)

        # Call SequencePredictor with arguments
        predicted_image_content, predicted_image_context, predicted_text_logits_k, h0_dec, c0_dec, _, _, _ = model(
            frames,
            input_ids_roberta_val,
            attention_mask_roberta_val,
            decoder_input_lstm_val
        )

        figure, ax = plt.subplots(2, 6, figsize=(20, 5), gridspec_kw={'height_ratios': [2, 1.5]})

        # Display description and image for the input frames, as text_dict_val contains only its text
        caption = tokenizer.decode(text_dict_val['input_ids'][0], skip_special_tokens=True)
        for i in range(4):
            im = frames[0, i, :, :, :].cpu()
            show_image(ax[0, i], im)
            ax[0, i].set_aspect('auto')
            ax[0, i].axis('off')
            wrapped_text = textwrap.fill(caption, width=40)
            ax[1, i].text(
                0.5, 0.99,
                wrapped_text,
                ha='center',
                va='top',
                fontsize=10,
                wrap=True
            )
            ax[1, i].axis('off')  # Hide axes for the text subplot

        # Display target image
        show_image(ax[0, 4], image_target_val[0].cpu())
        ax[0, 4].set_title('Target')
        ax[0, 4].set_aspect('auto')
        ax[0, 4].axis('off')

        # Display target text
        wrapped_text = textwrap.fill(tokenizer.decode(target_ids_val[0], skip_special_tokens=True), width=40)
        ax[1, 4].text(
            0.5, 0.99,
            wrapped_text,
            ha='center',
            va='top',
            fontsize=10,
            wrap=False)
        ax[1, 4].axis('off')

        # Display predicted image
        output = predicted_image_context[0, :, :, :].cpu()
        show_image(ax[0, 5], output)
        ax[0, 5].set_title('Predicted')
        ax[0, 5].set_aspect('auto')
        ax[0, 5].axis('off')

        # Generate and display predicted text (using h0_dec and c0_dec from model output)
        generated_tokens = generate(model.text_decoder,
                                    h0_dec[:, 0, :].unsqueeze(1),
                                    c0_dec[:, 0, :].unsqueeze(1),
                                    max_len=150,
                                    sos_token_id=tokenizer.cls_token_id,
                                    eos_token_id=tokenizer.sep_token_id)

        wrapped_text = textwrap.fill(tokenizer.decode(generated_tokens), width=40)

        ax[1, 5].text(
            0.5, 0.99,
            wrapped_text,
            ha='center',
            va='top',
            fontsize=10,
            wrap=False)
        ax[1, 5].axis('off')
        plt.tight_layout()
        plt.show()