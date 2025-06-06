import json
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


class QuestionAtomizer:
    def __init__(self, openai_api_key):
        self.chat_model = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=openai_api_key
        )

    def atomize_question(self, question):

        prompt = f"""
        Please decompose the following question into atomic sub-questions. 
        If the question cannot be further decomposed, simplify it using placeholders.
        Please return the sub-questions in JSON format as a list of strings.
        
        Here is an example that can be decomposed:
        Question: Whose father created the mechanical sheep?
        Sub-questions: ["[X] is the father of [Y]", "[Y] created the mechanical sheep]"]
        
        Here is two example that cannot be decomposed:
        Question: What movies did Albert Dupontel write?
        Sub-questions: ["Albert Dupontel wrote [X]"]
        Question: The movie Manson, when was it released?
        Sub-questions: ["Manson was released in [X]"]
        
        Question: {question}
        Sub-questions: 
        """

        messages = [HumanMessage(content=prompt)]
        response = self.chat_model.invoke(messages)

        try:
            sub_questions = json.loads(response.content)
            return sub_questions
        except json.JSONDecodeError:
            print(f"Error parsing response for question: {question}")
            return [question]


def process_dataset(input_file, output_file, atomizer):
    with open(input_file, 'r', encoding='utf-8') as f:
        data_point = json.load(f)

    processed_data = []
    for item in data_point:
        sub_questions = atomizer.atomize_question(item['question'])
        processed_item = {
            'question': item['question'],
            'answer': item['answer'],
            'sub_questions': sub_questions
        }
        processed_data.append(processed_item)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


def main(data):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    atomizer = QuestionAtomizer(openai_api_key)

    datasets = ['train', 'valid', 'test']
    for dataset in datasets:
        input_file = f"./raw/question/{data}/{dataset}.json"
        output_file = f"./processed/subquestion/{data}/{dataset}.json"

        print(f"Processing {dataset} dataset...")
        process_dataset(input_file, output_file, atomizer)
        print(f"Finished processing {dataset} dataset")


if __name__ == "__main__":

    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_API_BASE"] = ""

    data = "MovieQA"

    processed_path = f"./processed/subquestion/{data}"
    os.makedirs(processed_path, exist_ok=True)

    main(data)
