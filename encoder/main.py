from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description='Training and Testing Knowledge Graph Embedding Models')

    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)  # ['RotatE', 'pRotatE', 'DistMult', 'ComplEx', 'TransE']
    parser.add_argument('--max_steps', type=int)  # epoch = (max_steps*batch_size) // train_instance_num
    parser.add_argument('-d', '--hidden_dim', default=1024, type=int)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--get_train_candidates', type=str, default=True)
    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true', default=True)
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true', default=True)
    parser.add_argument('-n', '--negative_sample_size', default=2, type=int)
    parser.add_argument('-g', '--gamma', default=2.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--uni_weight', action='store_true', help='Otherwise use subsampling weighting like in word2vec', default=False)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    parser.add_argument('--log_steps', default=10, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=500, type=int, help='valid/test log every xx steps')

    parser.add_argument('--train_instance_num', type=int, help='DO NOT MANUALLY SET')
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(arguments)


def override_config(arguments):
    """
    Override model and data configuration
    """
    with open(os.path.join(arguments.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    arguments.countries = argparse_dict['countries']
    if arguments.data_path is None:
        arguments.data_path = argparse_dict['data_path']
    arguments.model = argparse_dict['model']
    arguments.double_entity_embedding = argparse_dict['double_entity_embedding']
    arguments.double_relation_embedding = argparse_dict['double_relation_embedding']
    arguments.hidden_dim = argparse_dict['hidden_dim']
    arguments.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, arguments):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """
    argparse_dict = vars(arguments)

    with open(os.path.join(arguments.save_path, f'config_step{arguments.max_steps}.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(arguments.save_path, f'checkpoint_step{arguments.max_steps}')
    )

    torch.save(model.state_dict(), os.path.join(arguments.save_path, f'{arguments.data}_{arguments.model}_step{arguments.max_steps}.pth'))
    torch.save(model, os.path.join(arguments.save_path, f'{arguments.data}_{arguments.model}_step{arguments.max_steps}-all.pth'))

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(os.path.join(arguments.save_path, f'{arguments.data}_{arguments.model}_ent_embeds_step{arguments.max_steps}'), entity_embedding)

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(os.path.join(arguments.save_path, f'{arguments.data}_{arguments.model}_rel_embeds_step{arguments.max_steps}'), relation_embedding)


def read_triple2id(file_path):
    """
    Read triples with ids. Format: head \t relation \t tail
    """
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((int(h), int(r), int(t)))
    return triples


def set_logger(arguments):
    """
    Write logs to checkpoint and console
    """
    if arguments.do_train:
        log_file = os.path.join(arguments.save_path or arguments.init_checkpoint, f'train_step{arguments.max_steps}.log')
    else:
        log_file = os.path.join(arguments.save_path or arguments.init_checkpoint, f'test_step{arguments.max_steps}.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs
    """
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def run(arguments):
    if (not arguments.do_train) and (not arguments.do_valid) and (not arguments.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if arguments.init_checkpoint:
        override_config(arguments)
    elif arguments.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if arguments.do_train and arguments.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if arguments.save_path and not os.path.exists(arguments.save_path):
        os.makedirs(arguments.save_path)

    # Write logs to checkpoint and console
    set_logger(arguments)

    with open(os.path.join(arguments.data_path, 'ent2id.txt'), encoding='utf-8') as fin:
        entity2id = dict()
        for line in fin:
            e_id, e_name = line.strip().split('\t')
            entity2id[e_name] = int(e_id)

    with open(os.path.join(arguments.data_path, 'rel2id.txt'), encoding='utf-8') as fin:
        relation2id = dict()
        for line in fin:
            r_id, r_name = line.strip().split('\t')
            relation2id[r_name] = int(r_id)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    arguments.nentity = nentity
    arguments.nrelation = nrelation

    logging.info('Model: %s' % arguments.model)
    logging.info('Data Path: %s' % arguments.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple2id(os.path.join(arguments.data_path, 'KG2id.txt'))
    args.train_instance_num = len(train_triples)
    logging.info('#train: %d' % len(train_triples))

    # All true triples
    all_true_triples = train_triples

    kge_model = KGEModel(
        model_name=arguments.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=arguments.hidden_dim,
        gamma=arguments.gamma,
        double_entity_embedding=arguments.double_entity_embedding,
        double_relation_embedding=arguments.double_relation_embedding
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if arguments.cuda:
        kge_model = kge_model.cuda()

    if arguments.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, arguments.negative_sample_size, 'head-batch'),
            batch_size=arguments.batch_size,
            shuffle=True,
            num_workers=max(1, arguments.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, arguments.negative_sample_size, 'tail-batch'),
            batch_size=arguments.batch_size,
            shuffle=True,
            num_workers=max(1, arguments.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = arguments.learning_rate
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, kge_model.parameters()), lr=current_learning_rate)
        if arguments.warm_up_steps:
            warm_up_steps = arguments.warm_up_steps
        else:
            warm_up_steps = arguments.max_steps // 2

    if arguments.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % arguments.init_checkpoint)
        checkpoint = torch.load(os.path.join(arguments.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if arguments.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Randomly Initializing %s Model...' % arguments.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % arguments.batch_size)
    logging.info('negative_adversarial_sampling = %d' % arguments.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % arguments.hidden_dim)
    logging.info('gamma = %f' % arguments.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(arguments.negative_adversarial_sampling))
    if arguments.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % arguments.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training
    if arguments.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        # 记录训练loss，先清空文件
        loss_path = os.path.join(arguments.save_path, f'loss_step{arguments.max_steps}.txt')
        with open(loss_path, 'w') as f:
            pass

        def loss_saver(loss_path, loss_item, step, arguments):
            with open(loss_path, 'a') as f_loss:
                f_loss.write(f"Epoch{step * arguments.batch_size // arguments.train_instance_num}, step{step}: {str(loss_item)}" + '\n')

        # Training loops
        training_logs = []
        for step in range(init_step + 1, arguments.max_steps + 1):
            log = kge_model.train_step(kge_model, optimizer, train_iterator, arguments)
            loss_saver(loss_path, log['loss'], step, arguments)  # 某一个step的loss
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % arguments.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, arguments)

            if step % arguments.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

        print("Saving final model...")
        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, arguments)

    # evaluation
    metrics = kge_model.test_step_raw(kge_model, train_triples, all_true_triples, arguments)
    log_metrics('Train', step, metrics)
    with open(os.path.join(arguments.save_path, f'result_train_step{arguments.max_steps}.txt'), 'w') as f:
        for metric in metrics:
            f.write('%s: %f\n' % (metric, metrics[metric]))


if __name__ == '__main__':
    args = parse_args()

    args.data = "MovieQA"
    args.max_steps = 5000  # max_steps = epoch * train_instance_num // batch_size
    args.model = "TransE"
    args.batch_size = 512

    args.data_path = f"./dataset/{args.data}"
    args.save_path = f"./saver/{args.data}/{args.data}_{args.model}"

    if args.model == 'RotatE':
        args.double_relation_embedding = False

    run(args)
