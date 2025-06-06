import json
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import numpy as np
from sentence_transformers import SentenceTransformer


def load_entities_relations(ent_file, rel_file):
    entities = []
    relations = []

    with open(ent_file, 'r', encoding='utf-8') as f:
        for line in f:
            _, entity = line.strip().split('\t')
            entities.append(entity)

    with open(rel_file, 'r', encoding='utf-8') as f:
        for line in f:
            _, relation = line.strip().split('\t')
            relations.append(relation)

    return entities, relations


class EntRelMatcher:
    def __init__(self,
                 openai_api_key,
                 model_path='./plm/deepset/sentence_bert',  # downloaded model path
                 top_k=20):
        self.chat_model = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.plm = SentenceTransformer(model_path)
        self.top_k = top_k

        self.entities = []
        self.relations = []
        self.entity_embeddings = None
        self.relation_embeddings = None

    def set_entities_relations(self, entities, relations):
        self.entities = entities
        self.relations = relations

        # calculate embeddings
        self.entity_embeddings = self.plm.encode(self.entities, show_progress_bar=True)
        self.relation_embeddings = self.plm.encode(self.relations, show_progress_bar=True)

    def match(self, sub_question):

        # embedding of sub-question
        question_embedding = self.plm.encode([sub_question])[0]

        # similarities between sub-question and entities
        entity_similarities = np.dot(self.entity_embeddings, question_embedding)
        top_k_entities = set(np.array(self.entities)[np.argsort(entity_similarities)[-self.top_k:]])

        # similarities between sub-question and relations
        relation_similarities = np.dot(self.relation_embeddings, question_embedding)
        top_k_relations = set(np.array(self.relations)[np.argsort(relation_similarities)[-self.top_k:]])

        # exact match using LLMs
        prompt = f"""
        Given the following sub-question and candidate entities/relations, please select the most relevant ones.
        There are two cases to handle:

        1. If the sub-question contains only one placeholder [X], it means we need to identify:
           - One explicit entity from the candidate entities
           - One explicit relation from the candidate relations

        2. If the sub-question contains two placeholders [X] and [Y], it means we only need to identify:
           - One explicit relation from the candidate relations
           - No explicit entities are needed in this case

        Sub-question: {sub_question}
        Candidate entities: {', '.join(top_k_entities)}
        Candidate relations: {', '.join(top_k_relations)}
        
        Please return the result in JSON format:
        {{
            "entities": ["entity1"],  # Only include if there's one placeholder
            "relations": ["relation1"],
        }}
        
        Examples:
        1. Sub-question: "[X] is the release date of The Matrix?"
           Response: {{
               "entities": ["The Matrix"],
               "relations": ["release_date"],
           }}

        2. Sub-question: "[X] is the father of [Y]"
           Response: {{
               "entities": [],
               "relations": ["father_of"],
           }}
        """

        messages = [HumanMessage(content=prompt)]
        response = self.chat_model.invoke(messages)

        try:
            result = json.loads(response.content)
            return set(result.get("entities", [])), set(result.get("relations", []))
        except json.JSONDecodeError:
            print(f"Error parsing response for sub-question: {sub_question}")
            return set(), set()


def process_dataset(input_file, output_file, matcher: EntRelMatcher):
    with open(input_file, 'r', encoding='utf-8') as f:
        data_point = json.load(f)

    processed_data = []
    for item in data_point:
        sub_questions = item['sub_questions']
        matched_entities = set()
        matched_relations = set()
        
        for sub_q in sub_questions:
            entities, relations = matcher.match(sub_q)
            matched_entities.update(entities)
            matched_relations.update(relations)
        
        if not matched_entities:
            print(f"Question \"{item['question']}\" has NULL topic_entities.")
        if not matched_relations:
            print(f"Question \"{item['question']}\" has NULL topic_relations.")
        
        processed_item = {
            'question': item['question'],
            'answer': item['answer'],
            'sub_questions': sub_questions,
            'topic_entities': list(matched_entities),
            'topic_relations': list(matched_relations)
        }
        processed_data.append(processed_item)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


def main(data):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    matcher = EntRelMatcher(openai_api_key)

    entities, relations = load_entities_relations(
        ent_file=f"./raw/kg/{data}/ent2id.txt",
        rel_file=f"./raw/kg/{data}/rel2id.txt"
    )
    matcher.set_entities_relations(entities, relations)

    datasets = ['train', 'valid', 'test']
    for dataset in datasets:
        input_file = f"./processed/subquestion/{data}/{dataset}.json"
        output_file = f"./processed/ent_rel_match/{data}/{dataset}.json"

        print(f"Processing {dataset} dataset...")
        process_dataset(input_file, output_file, matcher)
        print(f"Finished processing {dataset} dataset")


if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_API_BASE"] = ""

    data = "MovieQA"

    processed_path = f"./processed/ent_rel_match/{data}/"
    os.makedirs(processed_path, exist_ok=True)

    main(data)
