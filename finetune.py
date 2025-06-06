import os
import sys
import torch
import torch.cuda
import transformers
from datasets import load_dataset
from bridge import Knowledge_Bridge
from utils.tools import load_pretrain_embeddings, compute_mean_embeds

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_API_KEY"] = ""


def train(args):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Qwen2-LoRA model with params:\n"
            f"base_model: {args.llm_path}\n"
            f"train_data: {args.train_data}\n"
            f"lora_dir: {args.lora_dir}\n"
            f"ent_emb_dir: {args.ent_emb_dir}\n"
            f"ent_emb_dir: {args.rel_emb_dir}\n"
            f"num_epochs: {args.num_epochs}\n"
            f"batch_size: {args.batch_size}\n"
            f"micro_batch_size: {args.micro_batch_size}\n"
            f"learning_rate: {args.learning_rate}\n"
            f"cutoff_len: {args.cutoff_len}\n"
            f"val_set_size: {args.val_set_size}\n"
            f"num_prefix: {args.num_prefix}\n"
            f"lora_rank: {args.lora_rank}\n"
            f"lora_alpha: {args.lora_alpha}\n"
            f"lora_dropout: {args.lora_dropout}\n"
            f"lora_target_modules: {args.lora_target_modules}\n"
            f"train_on_inputs: {args.train_on_inputs}\n"
            f"add_eos_token: {args.add_eos_token}\n"
            f"group_by_length: {args.group_by_length}\n"
            f"wandb_project: {args.wandb_project}\n"
            f"wandb_run_name: {args.wandb_run_name}\n"
            f"wandb_watch: {args.wandb_watch}\n"
            f"wandb_log_model: {args.wandb_log_model}\n"
            f"resume_from_checkpoint: {args.resume_from_checkpoint}\n"
            f"prompt template: {args.prompt_template_name}\n"
        )

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    prompter = Prompter(args.prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(
            os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        textual_triples = ",".join([f"({h}, {r}, {t})" for h, r, t in data_point['retrieved_triples']]) + ". "

        full_prompt = prompter.generate_prompt(
            data_point['question'],
            textual_triples,
            data_point['answer'],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        tokenized_full_prompt['question_id'] = data_point['id']

        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=args.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",  # bias
        task_type="CAUSAL_LM",  # next word prediction
    )

    model = get_peft_model(model, config)  # add config

    # load training data
    if args.train_data.endswith(".json") or args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=args.train_data)
    else:
        data = load_dataset(args.train_data)

    # load KG embeddings and perform MEAN Pooling
    ent_embs, rel_embs = load_pretrain_embeddings(args.ent_emb_dir, args.rel_emb_dir)
    mean_ents_embeds, mean_rels_embeds = compute_mean_embeds(data['train']['retrieved_triples2id'], ent_embs, rel_embs)

    # knowledge bridge
    ft_model = Knowledge_Bridge(model, args.num_prefix, mean_ents_embeds, mean_rels_embeds, args.llm_hidden_size)

    # load checkpoint
    if args.resume_from_checkpoint:
        checkpoint_name = os.path.join(args.resume_from_checkpoint, "pytorch_model.bin")  # Full model checkpoint："{resume_from_checkpoint}/pytorch_model.bin"

        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")  # only LoRA model: "{resume_from_checkpoint}/adapter_model.bin"
            args.resume_from_checkpoint = False

        if os.path.exists(checkpoint_name):
            print(f"Restart training from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # model.print_trainable_parameters()

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=123)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        valid_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        print("Data Mapping...")
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        valid_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=ft_model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",  # Qwen2:adamw_hf, Qwen3:adamw_torch
            save_strategy="no",
            save_steps=0,
            save_total_limit=0,
            eval_strategy="steps" if args.val_set_size > 0 else "no",
            save_safetensors=False,
            eval_steps=None,
            output_dir=args.lora_dir,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=args.group_by_length,
            report_to=None,
            run_name="",
            disable_tqdm=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict

    # only for Qwen2
    if "Qwen2" in args.llm_path:
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if not os.path.exists(args.lora_dir):
        os.makedirs(args.lora_dir)

    if "Qwen2" in args.llm_path:
        model.save_pretrained(args.lora_dir, state_dict=old_state_dict())
        torch.save(ft_model.embeddings, os.path.join(args.lora_dir, "embeddings.pth"))

    elif "Qwen3" in args.llm_path:
        model.save_pretrained(
            args.lora_dir,
            safe_serialization=True,
            save_only_adapter=True
        )
        torch.save(ft_model.embeddings, os.path.join(args.lora_dir, "embeddings.pth"))
