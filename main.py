import time
import argparse
from finetune import train
from inference import test
from evaluation import evaluate
from utils.tools import clear_content, loss_statistic

import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)


def build_arg():
    parser = argparse.ArgumentParser(description='Generate answer for KGQA.')

    # dataset params
    parser.add_argument('--data', type=str, help='name of dataset')
    parser.add_argument('--train_data', type=str, help='path of training dataset')
    parser.add_argument('--valid_data', type=str, help='path of validation dataset')
    parser.add_argument('--test_data', type=str, help='path of test dataset')
    parser.add_argument('--response_path', type=str, help='path of response file')
    parser.add_argument('--result_path', type=str, help='path of result file')
    parser.add_argument('--evaluation_path', type=str, help='path of evaluation file')

    # embedding params
    parser.add_argument('--train_encoder_steps', type=int, help='number of training steps for encoder')
    parser.add_argument('--train_encoder_model', type=str, help='model to train encoder')
    parser.add_argument('--ent_emb_dir', type=str, help='directory of entity embedding')
    parser.add_argument('--rel_emb_dir', type=str, help='directory of relation embedding')

    # model paths
    parser.add_argument('--lora_dir', type=str, help='directory of lora weights')
    parser.add_argument('--llm_path', type=str, help='Path of LLM', default='./llm/Qwen/Qwen3-8B-Base')

    # lora params
    parser.add_argument('--lora_rank', type=int, help='rank of lora')
    parser.add_argument('--lora_alpha', type=int, help='alpha of lora')
    parser.add_argument('--lora_dropout', type=float, help='dropout of lora', default=0.05)
    parser.add_argument('--lora_target_modules', help='target modules of lora', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # llm hyperparams
    parser.add_argument('--llm_hidden_size', type=int, help='hidden size of llm', default=4096)
    parser.add_argument('--train_on_inputs', help='train on inputs', default=True)
    parser.add_argument('--add_eos_token', help='whether ass eos_token in input and output', default=False)
    parser.add_argument('--group_by_length', help='whether group by length, for speed up', default=True)

    # training hyperparams
    parser.add_argument('--num_epochs', type=int, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, help='batch size for training')
    parser.add_argument('--micro_batch_size', type=int, help='micro batch size for training', default=2)
    parser.add_argument('--val_set_size', type=float, help='validation set size', default=1)
    parser.add_argument('--learning_rate', type=float, help='learning rate for training', default=2e-5)
    parser.add_argument('--cutoff_len', type=float, help='cutoff length for training', default=512)

    # weight and bias (wandb) params
    parser.add_argument('--wandb_project', type=str, help='project name for wandb', default='')
    parser.add_argument('--wandb_run_name', type=str, help='run name for wandb', default='')
    parser.add_argument('--wandb_watch', type=str, help='watch model for wandb', default='')
    parser.add_argument('--wandb_log_model', type=str, help='logger model for wandb', default='')

    # additional params
    parser.add_argument('--num_prefix', type=int, help='number of prefix', default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, help='resume from checkpoint', default=None)
    parser.add_argument('--prompt_template_name', type=str, help='prompt template name')
    parser.add_argument('--run_mode', type=str, help='train or just test', default="train")
    parser.add_argument('--max_new_tokens', type=int, help='max new tokens for generation', default=512)
    parser.add_argument('--record_loss', type=bool, help='record loss', default=False)

    arg = parser.parse_args()

    return arg


if __name__ == '__main__':
    args = build_arg()

    # ==============================
    args.data = "MovieQA"
    args.num_epochs = 30
    args.train_encoder_steps = 5000
    args.train_encoder_model = "TransE"
    args.batch_size = 8
    args.lora_rank = 64
    # ==============================

    args.train_data = f"./data/{args.data}/train.json"
    args.valid_data = f"./data/{args.data}/valid.json"
    args.test_data = f"./data/{args.data}/test.json"
    args.ent_emb_dir = (f"./encoder/saver/{args.data}/{args.data}_{args.train_encoder_model}/"
                        f"{args.data}_{args.train_encoder_model}_ent_embeds_step{args.train_encoder_steps}.npy")
    args.rel_emb_dir = (f"./encoder/saver/{args.data}/{args.data}_{args.train_encoder_model}/"
                        f"{args.data}_{args.train_encoder_model}_rel_embeds_step{args.train_encoder_steps}.npy")
    args.lora_dir = f"./lora/{args.data}/run_epoch{args.num_epochs}"
    args.lora_alpha = args.lora_rank * 2
    args.prompt_template_name = args.data

    args.response_path = f"./logger/result/{args.data}/response_epoch{args.num_epochs}.txt"
    args.result_path = f"./logger/result/{args.data}/result_epoch{args.num_epochs}.json"
    args.evaluation_path = f"./logger/result/{args.data}/eval_epoch{args.num_epochs}.txt"

    # start training
    if args.run_mode == "train":
        if args.record_loss:
            clear_content(args)
        start_time = time.time()

        train(args)

        end_time = time.time()
        total_time = (end_time - start_time) / 3600
        print(f"Total time taken: {format(total_time, '.2f')} hours\n")

        if args.record_loss:
            os.makedirs(f"logger/loss/{args.data}", exist_ok=True)
            loss_statistic(args)

        # testing and evaluation
        os.makedirs(f"./logger/result/{args.data}", exist_ok=True)
        test(args)
        evaluate(args.result_path, args.evaluation_path)

    elif args.run_mode == "test":
        os.makedirs(f"./logger/result/{args.data}", exist_ok=True)

        test(args)
        evaluate(args.result_path, args.evaluation_path)

    else:
        print("Please input the correct run mode, should be 'train' or 'test'!\n")
