"""
Bangla TrOCR Training Script (Step 03)
Fine-tunes a Vision-Encoder-Decoder model on the prepared word-crop dataset.
"""

import pandas as pd
from datasets import Dataset
from PIL import Image
from transformers import (
    ViTImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import evaluate
import argparse
import os

# Define globally for metrics compute
processor = None
tokenizer = None

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Replace -100 in the labels as we can't decode them
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=pred_str, references=labels_str)
    return {"cer": cer}


def train_trocr(
    csv_path: str,
    output_dir: str = "weights/trocr_bangla",
    epochs: int = 5,
    batch_size: int = 8
):
    global processor, tokenizer
    print("Loading ViT encoder and Bangla BERT decoder...")
    
    # Use ViT for vision encoder and Bangla BERT for text decoder
    encoder_id = "google/vit-base-patch16-224-in21k"
    decoder_id = "sagorsarker/bangla-bert-base"
    
    processor = ViTImageProcessor.from_pretrained(encoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)
    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id)
    
    # Configure model parameters for TrOCR
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Generation parameters (must be in generation_config for new huggingface versions)
    model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.max_length = 64
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = hf_dataset['train']
    eval_ds = hf_dataset['test']

    def preprocess_batch(batch):
        # Read images
        images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        
        # Tokenize text natively in Bangla
        labels = tokenizer(
            batch["text"], 
            padding="max_length", 
            max_length=64, 
            truncation=True
        ).input_ids
        
        # Replace pad token with -100 to ignore in loss
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        
        batch["pixel_values"] = pixel_values
        batch["labels"] = labels
        return batch

    print("Preprocessing training data...")
    train_ds.set_transform(preprocess_batch)
    print("Preprocessing evaluation data...")
    eval_ds.set_transform(preprocess_batch)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        eval_strategy="epoch",
        remove_unused_columns=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=False, 
        logging_steps=50,
        save_strategy="epoch",
        num_train_epochs=epochs,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    print("Starting TrOCR Fine-tuning...")
    trainer.train()
    
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/trocr_dataset/train_manifest.csv")
    parser.add_argument("--output", type=str, default="weights/trocr_bangla")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    
    train_trocr(args.csv, args.output, args.epochs, args.batch)
