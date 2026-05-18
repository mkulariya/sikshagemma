from lm_eval.api.task import Task
from lm_eval.api.registry import TASK_REGISTRY
from lm_eval.api.utils import eval_logger
from datasets import load_dataset
import json

@TASK_REGISTRY.register("reja1_jeeneet")
class Reja1JEENEET(Task):
    VERSION = 0
    DATASET_PATH = "Reja1/jee-neet-benchmark"
    DATASET_NAME = None
    SPLIT = "test"

    def __init__(self, config=None):
        super().__init__(config=config)
        self.dataset = None

    def download(self, data_dir=None):
        self.dataset = load_dataset(self.DATASET_PATH, split=self.SPLIT)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.dataset is None:
            self.download()
        return self.dataset

    def process_docs(self, dataset):
        def _process(item):
            correct = item["correct_answer"]
            try:
                correct_parsed = json.loads(correct)
                if isinstance(correct_parsed, list):
                    item["answer"] = ",".join(str(a) for a in correct_parsed)
                else:
                    item["answer"] = str(correct_parsed)
            except (json.JSONDecodeError, TypeError):
                item["answer"] = str(correct).strip()
            return item
        return dataset.map(_process)

    def doc_to_text(self, doc):
        return "Look at this exam question image. What is the correct answer? Provide just the answer letter or number."

    def doc_to_target(self, doc):
        return doc["answer"]

    def build_prompt(self, doc):
        # Multimodal: returns a dict with image and text
        return {
            "image": doc["image"],
            "text": self.doc_to_text(doc)
        }

    def construct_requests(self, doc, ctx):
        return [{
            "type": "generate_until",
            "arguments": {
                "until": ["\n", ".", " ", "Question"],
                "do_sample": False,
                "temperature": 0.0
            }
        }]

    def process_results(self, doc, results):
        prediction = results[0].strip().upper()
        target = doc["answer"].strip().upper()
        exact_match = prediction == target
        return {"exact_match": exact_match}

    def aggregation(self):
        return {"exact_match": "mean"}

    def higher_is_better(self):
        return {"exact_match": True}