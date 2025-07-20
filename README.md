# Fast & Efficient LLM Fine-tuning with LoRA for Spam Classification

This project demonstrates a **parameter-efficient fine-tuning** approach for adapting pretrained **BERT** models to a downstream **spam classification** task. It compares full model fine-tuning, classifier-head tuning, and **Low-Rank Adaptation (LoRA)** to showcase trade-offs in performance, training cost, and efficiency.

Built with PyTorch and HuggingFace Transformers, this notebook manually implements LoRA by injecting low-rank adapters into BERT's self-attention layers, This enables fine-tuning **just 1.3M parameters**—only **1.2% of the full model (109M)**—while retaining **over 99.5%** of the full-model accuracy and reducing training time by **33%**.

## Features

- **LLM Fine-tuning Strategies**: Implements and compares:
  - Full fine-tuning (all model weights)
  - Classifier-head only tuning
  - Parameter-efficient LoRA tuning
- **LoRA from Scratch**: Manually implements LoRA by modifying the self-attention mechanism inside BERT's architecture using low-rank adapter matrices.
- **Performance Analysis**: Evaluates training time, model size, and accuracy across different methods.
- **Efficient Training**: LoRA achieves near-identical performance with substantially fewer parameters and faster convergence.


## Dataset

- The [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from UCI is used.
- It consists of 5,574 English SMS messages labeled as "spam" or "ham".


## Usage
Open the notebook and run the notebook step-by-step. You will:
- Load and tokenize SMS data
- Apply LoRA by injecting low-rank adapters directly into attention layers
- Fine-tune BERT using 3 strategies
- Evaluate accuracy, parameter count, and training time
