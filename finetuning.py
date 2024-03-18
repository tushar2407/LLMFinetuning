base_model = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-2-6k"

def training_function():
    from accelerate import Accelerator, DeepSpeedPlugin
    import os
    import deepspeed
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging, get_linear_schedule_with_warmup
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer
    from torch.utils.data import Dataset, DataLoader
    import datasets
    import transformers
    from tqdm import tqdm
    
    
    dataset = load_dataset('json', data_files='6k_train.json')

    def combine_features(examples):
        combined_text = "Instruction: " + examples['instruction'] + " Question: " + examples['question'] + " Answer: " + examples['answer']
        return {"combined_text": combined_text}

    dataset['train'] = dataset['train'].map(combine_features)

    train_dataset = dataset["train"].map(lambda examples: {"text": examples["combined_text"]}, remove_columns=dataset["train"].column_names)
    train_dataset.set_format(type="torch", columns=["text"])

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    use_flash_attention = False

    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        device_map="auto",
        quantization_config=bnb_config,
        use_cache=False,
        use_flash_attention_2=use_flash_attention,
        torch_dtype=torch.float16
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1


        
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir="results",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=1e-5,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
        report_to="tensorboard",
        save_steps=100,
        save_total_limit=5,
    )

    max_seq_length = 1024  # max sequence length for model and packing of the dataset

    # model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_text_field="text",
        args=args,
    )

    trainer.train()

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

    return model

if __name__=='__main__':
    model = training_function()
