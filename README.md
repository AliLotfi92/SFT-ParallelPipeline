# Pipeline Parallelism for LLaMA Model Fine-Tuning

Implementation of pipeline parallelism is not hard. This repo provieds a simple code example for fine-tuning the LLaMA 3.2-1B Instruct model. Using this the model would be splitted across multiple GPUs to address memory limitations during training.

## Features
- **Pipeline Parallelism:** To address large model memory requirements, it splitts the model across multiple GPUs.
- **LoRA Integration:** Reduces memory usage and prevents overfitting.
- **DeepSpeed Optimization:** I leveraged DeepSpeed for distributed training and efficient memory management.

## Prerequisites
- Python 3.8+
- PyTorch
- Transformers
- Datasets
- DeepSpeed

## Installation
```bash
pip install requirement.txt
```

## Pipeline Components
### 1. `train.py`
- **Model and Tokenizer Loading:** You can use any model, but I used LlaMA 3.2-1B Instruct model.
- **Dataset Preparation:** Utilizes a custom `SummarizationDataset` class for preprocessing the XSum dataset, you may change this file for your custom dataset.

### 2. `dataset_.py`
Contains the `SummarizationDataset` class for preparing and preprocessing the XSum dataset.

- **Data Preprocessing:** Tokenizes the input and output text with padding and truncation.
- **Instruction Prefix:** Adds an instruction prompt to each document.
- **Attention Masks and Labels:** Proper attention (partially masked) and loss calculation by handling padding tokens and special token positions.

### 3. `deepspeed_config.json`
DeepSpeed configuration for optimizing training.

- **Zero Redundancy Optimizer (ZeRO) Stage 3:** This is the most useful stage of ZeRO optimization. It splits  all three critical factors of training—model states, gradients, and optimizer states across gpus. 
- **FP16 Training:** Automatically adjusts the loss scaling factor (upscaling/downscaling) to maintain numerical stability during training, especially for gradients.

## Usage
1. **Prepare Data and Model:**
   Ensure the dataset and model are correctly set up using the provided scripts.

2. **Run Training:**
   ```bash
   deepspeed --num_gpus=4 train.py 
   ```

3. **Save Model:**
   The finetuned model and tokenizer will be saved in the `./final_model` directory.

## Output
- **Training Logs:** Stored in the `./logs` directory.
- **Model Checkpoints:** Saved in the `./results` directory.
- **Final Model:** Saved in the `./final_model` directory.

## Notes
- The model is split across GPUs to handle large memory requirements. Adjust `per_device_train_batch_size` and `per_device_eval_batch_size` as needed based on GPU availability.
