import json
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.tools import load_pretrain_embeddings, compute_mean_embeds4test

from transformers.utils import logging

logging.get_logger("transformers").setLevel(logging.ERROR)


def load_test_dataset(path):
    t_dataset = json.load(open(path, "r", encoding="utf-8"))
    return t_dataset


def test(args):
    print("\nTest fine-tuned LLM\n")

    cuda = "cuda:0"

    prompter = json.load(open(f"./prompts/{args.prompt_template_name}.json", "r", encoding="utf-8"))
    prompt_template = prompter["prompt_input"]

    print(f'Loading tokenizer from {args.llm_path}...\n')
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, padding_side="left", trust_remote_code=True)

    ent_embeds, rel_embeds = load_pretrain_embeddings(args.ent_emb_dir, args.rel_emb_dir)

    if "Qwen2" in args.llm_path:
        print(f'Qwen2: Loading base model from {args.llm_path}...')
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f'Qwen2: Loading lora fine-tuned model from {args.llm_path}...')
        model = PeftModel.from_pretrained(
            model,
            args.lora_dir,
            torch_dtype=torch.bfloat16,
        )
    elif "Qwen3" in args.llm_path:
        print(f'Loading merged LoRA model from {args.llm_path}...\n')
        model = AutoModelForCausalLM.from_pretrained(
            args.lora_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.eval()
    else:
        assert False, "Unknown model type"

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # pad_token_id in finetune
    model.config.bos_token_id = 151643  # see llm_path/config.json
    model.config.eos_token_id = 151643  # see llm_path/config.json

    model = model.to(cuda)

    results = []
    responses = []
    for data in tqdm(load_test_dataset(args.test_data), desc='Testing...'):

        textual_triples = ",".join([f"({h}, {r}, {t})" for h, r, t in data['retrieved_triples']]) + ". "

        idx = data["id"]
        question = data["question"]
        ans = data["answer"]

        # soft prompt
        mean_ent_embs, mean_rel_embs = compute_mean_embeds4test(data['retrieved_triples2id'], ent_embeds, rel_embeds)
        kb_adapter = torch.load(f"{args.lora_dir}/embeddings.pth", weights_only=False).adapter.to(cuda)
        pretrain_embs = torch.stack((mean_ent_embs, mean_rel_embs), dim=1)
        prefix = kb_adapter(pretrain_embs)

        # hard prompt
        prompt = prompt_template.format(question=question, background=textual_triples)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(cuda)
        token_embeds = model.model.embed_tokens(input_ids)  # or: model.model.model.embed_tokens(input_ids)

        # knowledge concat
        input_embeds = torch.cat((prefix, token_embeds), dim=1)

        generate_ids = model.generate(
            inputs_embeds=input_embeds.to(dtype=torch.bfloat16),
            max_new_tokens=args.max_new_tokens,
        )

        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res = response.replace(context, "").strip()

        saved_response = f'INDEX {idx}:\n' + response + '\n\n'
        responses.append(saved_response)

        print(f'\n-------------------------------\nINDEX {idx}:\n' + str(response) + '\n-------------------------------\n')

        results.append(
            {
                "index": idx,
                "question": question,
                "answer": ans,
                "prediction": res
            }
        )

    print('Saving all responses...')
    with open(args.response_path, 'w', encoding='utf-8') as f2:
        f2.writelines(responses)

    print('Saving all predictions (results)...')
    with open(args.result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
