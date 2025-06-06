import torch
import torch.nn as nn
from typing import Optional, List

from transformers import Qwen3ForCausalLM


def llm_hidden_size(size):
    return size


class Knowledge_Bridge(nn.Module):
    def __init__(
            self,
            model: Qwen3ForCausalLM,
            num_prefix,
            ent_embs,  # mean value
            rel_embs,  # mean value
            hidden_size,  # LLM hidden size
            pretrain_emb_path: str = None  # pretrained bridger path
    ) -> None:
        super(Knowledge_Bridge, self).__init__()
        self.qwen3_model = model

        if pretrain_emb_path is None:
            print("\nKnowledge Bridger Trained From Scratch\n")
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=ent_embs,
                pretrain_rel_embs=rel_embs,
                dim_llm=hidden_size,
                num_prefix=num_prefix
            )
        else:
            print("Knowledge Bridger Load From {}".format(pretrain_emb_path))
            self.embeddings = torch.load(pretrain_emb_path)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            question_id: torch.LongTensor = None,
            # my_param: torch.LongTensor = None,
    ):
        # print(my_param)
        kg_embeds = self.embeddings(question_id)
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.qwen3_model.model.model.embed_tokens(input_ids)  # token embeddings
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)  # kg embeddings and text embeddings
        prefix_mask = torch.ones((batch_size, seq_len))  # prefix_mask for prefix
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)

        return self.qwen3_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds.to(torch.bfloat16),
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PretrainKGEmbedding(nn.Module):
    def __init__(
            self,
            pretrain_ent_embs,
            pretrain_rel_embs,
            dim_llm,
            num_prefix
    ):
        super(PretrainKGEmbedding, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = nn.Embedding.from_pretrained(pretrain_ent_embs)
        self.rel_embeddings = nn.Embedding.from_pretrained(pretrain_rel_embs)
        self.pretrain_dim = self.ent_embeddings.weight.shape[1]

        # Froze the pretrain embeddings
        self.ent_embeddings.requires_grad_(False)
        self.rel_embeddings.requires_grad_(False)
        self.adapter = nn.Linear(self.pretrain_dim, self.emb_dim)

    def forward(self, question_id):
        mean_ent_embs = self.ent_embeddings(question_id)
        mean_rel_embs = self.rel_embeddings(question_id)

        pretrain_embs = torch.stack((mean_ent_embs, mean_rel_embs), dim=1)
        prefix = self.adapter(pretrain_embs)

        return prefix
