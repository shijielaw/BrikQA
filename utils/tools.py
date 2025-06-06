import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_average_loss(file_path, num_epochs):
    with open(file_path, 'r') as file:
        losses = file.readlines()

    loss_values = [float(ls.strip()) for ls in losses]

    total_batches = len(loss_values)
    batches_per_epoch = total_batches // num_epochs

    epoch_losses = []
    for epoch in range(num_epochs):
        start_idx = epoch * batches_per_epoch
        end_idx = (epoch + 1) * batches_per_epoch
        epoch_loss_values = loss_values[start_idx:end_idx]
        average_loss = sum(epoch_loss_values) / len(epoch_loss_values) if epoch_loss_values else 0
        epoch_losses.append("epoch" + str(epoch + 1) + " loss: " + str(average_loss))

    return epoch_losses


def clear_content(argus):
    if torch.cuda.device_count() == 1:
        with open(f"logger/loss/{argus.data}/loss_singleGPU-{argus.num_epochs}epoch.txt", "w") as f1:
            pass
    elif torch.cuda.device_count() > 1:
        with open(f"logger/loss/{argus.data}/loss_multipleGPU-{argus.num_epochs}epoch.txt", "w") as f2:
            pass


def loss_statistic(argus):
    if torch.cuda.device_count() == 1:
        epoch_loss = calculate_average_loss(
            f'logger/loss/{argus.data}/loss_singleGPU-{argus.num_epochs}epoch.txt', argus.num_epochs)
        print(f"Average loss per epoch in single GPU:\n {epoch_loss} ")

    elif torch.cuda.device_count() > 1:
        epoch_loss = calculate_average_loss(
            f'logger/loss/{argus.data}/loss_multipleGPU-{argus.num_epochs}epoch.txt', argus.num_epochs)
        print(f"Average loss per epoch in multiple GPU:\n: {epoch_loss}")


def load_pretrain_embeddings(ent_emb_path, rel_emb_path):
    ent_embs = np.load(ent_emb_path)
    ent_embs = torch.tensor(ent_embs).to(device)
    ent_embs.requires_grad = False

    rel_embs = np.load(rel_emb_path)
    rel_embs = torch.tensor(rel_embs).to(device)
    rel_embs.requires_grad = False

    ent_dim = ent_embs.shape[1]
    rel_dim = rel_embs.shape[1]

    # print(ent_dim, rel_dim)

    if ent_dim != rel_dim:  # RotatE
        rel_embs = torch.cat((rel_embs, rel_embs), dim=-1)

    return ent_embs, rel_embs


def compute_mean_embeds(retrieved_triples2id, entity_embeds, relation_embeds):
    n_samples = len(retrieved_triples2id)
    embed_dim = entity_embeds.shape[1]

    mean_ents_matrix = torch.zeros((n_samples, embed_dim))
    mean_rels_matrix = torch.zeros((n_samples, embed_dim))

    for i, sample in enumerate(retrieved_triples2id):
        ent_ids = set()
        rel_ids = []

        for triple in sample:
            h, r, t = triple
            ent_ids.update([h, t])
            rel_ids.append(r)

        if ent_ids:
            mean_ents_matrix[i] = torch.mean(entity_embeds[list(ent_ids)], dim=0, keepdim=True)
        if rel_ids:
            mean_rels_matrix[i] = torch.mean(relation_embeds[rel_ids], dim=0, keepdim=True)

    return mean_ents_matrix, mean_rels_matrix


def compute_mean_embeds4test(retrieved_triples2id, entity_embeds, relation_embeds):
    n_samples = 1
    embed_dim = entity_embeds.shape[1]

    mean_ents_matrix = torch.zeros((n_samples, embed_dim))
    mean_rels_matrix = torch.zeros((n_samples, embed_dim))

    ent_ids = set()
    rel_ids = []
    for i, sample in enumerate(retrieved_triples2id):
        h, r, t = sample
        ent_ids.update([h, t])
        rel_ids.append(r)

    if ent_ids:
        mean_ents_matrix = torch.mean(entity_embeds[list(ent_ids)], dim=0, keepdim=True)
    if rel_ids:
        mean_rels_matrix = torch.mean(relation_embeds[rel_ids], dim=0, keepdim=True)

    return mean_ents_matrix, mean_rels_matrix
