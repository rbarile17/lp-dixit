from ..dataset import Dataset
from ..utils import write_json


class ExplanationBuilder:
    def __init__(self, dataset: Dataset, max_explanation_length: int):
        self.dataset = dataset
        self.candidates_path = None
        self.summarization = None

        self.length_cap = max_explanation_length

        self.start = None
        self.end = None

    def save_cands(self, i, cands):
        write_json(cands, self.candidates_path / f"{i}.json")

    def build_explanations(self, i, cands: list, k: int = 10):
        pass

    def process_output_cands(self, cands):
        output_cands = []
        if self.summarization_flag:
            for evaluated_cand in cands:
                cand = self.summarization.map_rule(evaluated_cand["cand"])
                cand = self.dataset.labels_triples(cand)
                qcand = self.label_qtriples(evaluated_cand["cand"])
                engine_output = evaluated_cand["engine_output"]
                output_cand = {
                    "qcand": qcand,
                    "cand": cand,
                    "engine_output": engine_output,
                }
                output_cands.append(output_cand)
        else:
            output_cands = cands
            for cand in output_cands:
                cand["cand"] = self.dataset.labels_triples(cand["cand"])

        return output_cands

    def format_output(self, pred, cands):
        best_cand = cands[0]
        output = {
            "pred": self.dataset.labels_triple(pred),
            "explanation": best_cand["cand"],
            "explanation_relevance": best_cand["engine_output"]["rel"],
            "#relevances": self.rels,
            "execution_time": self.end - self.start,
        }

        if "qcand" in best_cand:
            output["quotient_explanation"] = best_cand["qcand"]

        return output
