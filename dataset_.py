from datasets import load_dataset

class SummarizationDataset(object):

    def __init__(self, tokenizer, max_length=256):
        self.tokenize = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset('xsum', trust_remote_code=True)

        # get the data
        self.train_data = self.dataset['train'].shuffle(seed=42).select(range(1000))
        self.val_data = self.dataset['validation'].shuffle(seed=42).select(range(200))

    # preparing the data for instruction sft
    def preprocess_function(self, examples):
        # add instruction (summarize this document)
        inputs = ["Summarize this document: " + doc + "\n" for doc in examples['document']]
        # targets are summary of document created by human
        targets = [summary + self.tokenize.eos_token for summary in examples['summary']]

        # tokenizing input (instruction + documents) and outputs, pad to max_length
        tokenized_inputs = self.tokenize(
            inputs,
            max_length=self.max_length // 2,
            padding='max_length',
            truncation=True,
            return_tensors=None, 
        )
        # tokenizing target (summary) and outputs, pad to max_length   
        tokenized_targets = self.tokenize(
            targets,
            max_length=self.max_length // 2,
            padding='max_length',
            truncation=True,
            return_tensors=None,
        )

        # creating input: input = input + target[1:], the reason for target[1:] to not repeat beginning of sentence
        input_ids = [
            input_id + target_id[1:] 
            for input_id, target_id in zip(tokenized_inputs['input_ids'], tokenized_targets['input_ids'])
        ]
        # creating attention mask: attending to non-padding tokens
        attention_mask = [
            mask + [1] * (len(target_id) - 1)
            for mask, target_id in zip(tokenized_inputs['attention_mask'], tokenized_targets['input_ids'])
        ]

        # implicitly applying partial masking, we just compute the loss on target (summaries) not inputs
        labels = [
            [-100] * len(input_id) + target_id[1:]
            for input_id, target_id in zip(tokenized_inputs['input_ids'], tokenized_targets['input_ids'])
        ]

        # padding to have similar size tokens
        input_ids = [self.pad_or_truncate(seq, self.max_length, self.tokenize.pad_token_id) for seq in input_ids]
        attention_mask = [self.pad_or_truncate(seq, self.max_length, 0) for seq in attention_mask]
        labels = [self.pad_or_truncate(seq, self.max_length, -100) for seq in labels]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    # padding to have similar size tokens
    @staticmethod
    def pad_or_truncate(sequence, max_length, pad_value):
        if len(sequence) > max_length:
            return sequence[:max_length]
        elif len(sequence) < max_length:
            return sequence + [pad_value] * (max_length - len(sequence))
        return sequence

    def get_train_or_val(self, split="train"):
        if split == "train":
            train_dataset = self.train_data.map(
                self.preprocess_function,
                batched=True,
                remove_columns=self.train_data.column_names,
                desc="Running tokenizer on train dataset",
            )
            return train_dataset
        else:
            val_dataset = self.val_data.map(
                self.preprocess_function,
                batched=True,
                remove_columns=self.val_data.column_names,
                desc="Running tokenizer on validation dataset",
            )
            return val_dataset
