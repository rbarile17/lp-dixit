import time

from .explanation_builder import ExplanationBuilder

from .utils import sort_by_relevance
from .utils import log_cands

class CriageBuilder(ExplanationBuilder):
    def __init__(self, engine, reverse=False):
        super().__init__(engine.dataset, 1)

        self.reverse = reverse
        self.engine = engine

    def build_explanations(self, i, pred, cands: list):
        self.start = time.time()
        pred_s, _, _ = pred
        evaluated_cands = []

        for cand in cands:
            _, _, o = cand
            perspective = "head" if o == pred_s else "tail"
            engine_output = self.engine.compute_relevance(pred, cand, perspective)
            evaluated_cands.append({"cand": [cand], "engine_output": engine_output})

        self.rels = len(evaluated_cands)

        evaluated_cands = sort_by_relevance(evaluated_cands)
        evaluated_cands = self.process_output_cands(evaluated_cands)
        log_cands(i, evaluated_cands)
        self.save_cands(i, evaluated_cands)
        self.end = time.time()

        return self.format_output(pred, evaluated_cands)
