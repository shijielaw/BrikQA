import json
import os
import networkx as nx
from tqdm import tqdm


class SubgraphRetriever:
    def __init__(self, kg_path):
        self.kg_path = kg_path
        self.graph = nx.DiGraph()
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        self._load_kg()

    def _load_kg(self):
        with open(os.path.join(self.kg_path, 'ent2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                id, entity = line.strip().split('\t')
                self.entity2id[entity] = int(id)
                self.id2entity[int(id)] = entity

        with open(os.path.join(self.kg_path, 'rel2id.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                id, relation = line.strip().split('\t')
                self.relation2id[relation] = int(id)
                self.id2relation[int(id)] = relation

        with open(os.path.join(self.kg_path, 'KG.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                head, rel, tail = line.strip().split('\t')
                head_id = self.entity2id[head]
                rel_id = self.relation2id[rel]
                tail_id = self.entity2id[tail]

                self.graph.add_edge(head_id, tail_id, relation=rel_id)

    def get_subgraph(self, entities, relations, max_depth=2):
        entity_ids = {self.entity2id[e] for e in entities if e in self.entity2id}
        relation_ids = {self.relation2id[r] for r in relations if r in self.relation2id}

        triples = []
        triples2id = []

        # case1: both topic_entities and topic_relations are provided
        if entity_ids and relation_ids:
            for u, v, data in self.graph.edges(data=True):
                rel_id = data['relation']
                if rel_id in relation_ids and (u in entity_ids or v in entity_ids):
                    triples.append([
                        self.id2entity[u],
                        self.id2relation[rel_id],
                        self.id2entity[v]
                    ])
                    triples2id.append([u, rel_id, v])

        # case2: only topic_relations are provided
        elif relation_ids:
            for u, v, data in self.graph.edges(data=True):
                rel_id = data['relation']
                if rel_id in relation_ids:
                    triples.append([
                        self.id2entity[u],
                        self.id2relation[rel_id],
                        self.id2entity[v]
                    ])
                    triples2id.append([u, rel_id, v])

        return {
            "triples": triples,
            "triples2id": triples2id,
            "count": len(triples)
        }


def process_dataset(input_file, output_file, retriever):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for idx, item in enumerate(tqdm(data, desc="Processing questions")):
        matched_entities = set(item.get('topic_entities', []))
        matched_relations = set(item.get('topic_relations', []))

        subgraph = retriever.get_subgraph(matched_entities, matched_relations)

        processed_item = {
            "id": idx,
            "question": item['question'],
            "answer": item['answer'],
            "topic_entities": item.get('topic_entities', [])[0] if item.get('topic_entities') else "",
            "topic_relations": item.get('topic_relations', [])[0] if item.get('topic_relations') else "",
            "retrieved_triples": subgraph["triples"],
            "retrieved_triples2id": subgraph["triples2id"],
            "retrieved_triples_count": subgraph["count"]
        }

        processed_data.append(processed_item)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


def main(data: str):
    retriever = SubgraphRetriever(f"./raw/kg/{data}")

    datasets = ['train', 'valid', 'test']
    for dataset in datasets:
        input_file = f"./processed/ent_rel_match/{data}/{dataset}.json"
        output_file = f"./processed/subgraph/{data}/{dataset}.json"

        print(f"Processing {dataset} dataset...")
        process_dataset(input_file, output_file, retriever)
        print(f"Finished processing {dataset} dataset")


if __name__ == "__main__":
    data = "MovieQA"

    processed_path = f"./processed/subgraph/{data}/"
    os.makedirs(processed_path, exist_ok=True)

    main(data)
