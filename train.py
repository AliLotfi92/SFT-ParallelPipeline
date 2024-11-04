from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import math
from transformers.trainer_callback import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from dataset_ import SummarizationDataset
from transformers import DataCollatorForSeq2Seq 


# load your desire model here, here I used recently released Llama 3.2, 1B, Instruct
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./llama3.2-1B-instruct")
# since the llama modle does not have pad token, we can create with one with another special token to it
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

summary_dataset = SummarizationDataset(tokenizer=tokenizer)
train_dataset = summary_dataset.get_train_or_val(split="train")
val_dataset = summary_dataset.get_train_or_val(split="val")


train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# load the model here
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./llama3.2-1B-instruct",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
# adjust the model for pad tokens
model.config.pad_token_id = tokenizer.pad_token_id

# using lora to: 1- use less gpu memory, 2- prevent overfitting
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)


# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=10,
    evaluation_strategy="steps",
    logging_dir='./logs',
    logging_steps=10,
    save_steps=len(train_dataset) // 2,
    eval_steps=len(train_dataset) // 2,
    save_total_limit=3,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    deepspeed="deepspeed_config.json",
)

# data collator for dataloader
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

#
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)] #  Early Stopping if No improvement after 10 consecutive evaluation steps, the training will stop.
)

# Training
train_result = trainer.train()

# Print training results
print("\n Done!")

output_dir = "./final_model"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nModel and tokenizer saved to {output_dir}")
