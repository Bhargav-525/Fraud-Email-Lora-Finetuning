# Fraud Email Classification with LoRA Fine-Tuning

Fine-tuned an open-weight large language model to classify emails as phishing or legitimate, using parameter-efficient fine-tuning (LoRA).

## Overview

This project fine-tunes **Qwen2.5-0.5B**, an open-weight transformer-based LLM, for binary text classification (Phishing Email vs. Safe Email) using **LoRA (Low-Rank Adaptation)** via Hugging Face's `peft` library. Rather than updating all ~495M model parameters, LoRA trains a small set of adapter weights (~542K parameters, ~0.11% of the total), making fine-tuning fast and lightweight while still achieving strong task performance.

## Method

- **Base model:** Qwen/Qwen2.5-0.5B (Hugging Face Hub)
- **Fine-tuning technique:** LoRA (rank=8, alpha=16, dropout=0.05) targeting attention projection layers (`q_proj`, `v_proj`)
- **Task:** Binary sequence classification (Phishing Email = 0, Safe Email = 1)
- **Dataset:** Public phishing/email dataset from Hugging Face Hub (~18,650 labeled emails)
- **Training:** 2 epochs, learning rate 2e-4, evaluated on a held-out 15% validation split
- **Stack:** Python, PyTorch, Hugging Face `transformers`, `peft`, `datasets`, `evaluate`

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | 98.03% |
| Validation F1 (weighted) | 0.9804 |
| Validation Loss | 0.165 |

## Repository Contents

- `train_lora_fraud_classifier.ipynb` — full training notebook (data loading, preprocessing, LoRA setup, training loop, evaluation)
- `lora_fraud_classifier_adapter.zip` — saved LoRA adapter weights and tokenizer files from the final trained model

## Why LoRA

Full fine-tuning of even a small 0.5B-parameter LLM requires significant compute and memory. LoRA freezes the base model and injects small trainable low-rank matrices into selected layers, cutting trainable parameters by over 99% while retaining most of full fine-tuning's performance — making it practical to fine-tune on free-tier GPU resources (Kaggle T4).

## Notes

This project was built as applied practice in LLM fine-tuning techniques (SFT-adjacent classification fine-tuning, PEFT/LoRA), relevant to real-world use cases like fraud detection and content moderation.
