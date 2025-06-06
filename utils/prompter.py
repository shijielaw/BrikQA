import json
import os.path as osp


class Prompter(object):

    def __init__(self, template_name):

        file_name = osp.join("prompts", f"{template_name}.json")

        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")

        with open(file_name, "r", encoding="utf-8") as fp:
            self.template = json.load(fp)

    def generate_prompt(self, question, background, answer=None):

        res = self.template["prompt_input"].format(question=question, background=background)

        if answer:
            res = f"{res}{answer}"
        return res

