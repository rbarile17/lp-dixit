import itertools
import random
import time

from .explanation_builder import ExplanationBuilder
from .summarization import Simulation, Bisimulation

from .utils import log_cands
from .utils import sort_by_relevance, sort_by_relevance_and_length


class StochasticBuilder(ExplanationBuilder):
    def __init__(
        self, xsi, engine, summarization: str = None, max_explanation_length: int = 4
    ):
        dataset = engine.dataset
        super().__init__(dataset=dataset, max_explanation_length=max_explanation_length)

        self.window_size = 10

        self.xsi = xsi
        self.engine = engine

        self.rels = 0
        self.cand_to_rel = {}

        self.summarization = None
        if summarization == "simulation":
            self.summarization = Simulation(dataset)
        elif summarization == "bisimulation":
            self.summarization = Bisimulation(dataset)

    def label_qtriples(self, qtriples):
        label_qtriples = []
        for qtriple in qtriples:
            s_part, p, o_part = qtriple
            s_part = [self.dataset.id_to_entity[e] for e in s_part]
            o_part = [self.dataset.id_to_entity[e] for e in o_part]
            p = self.dataset.id_to_relation[p]
            label_qtriples.append((s_part, p, o_part))

        return label_qtriples

    def build_explanations(self, i, pred, cands: list):
        self.rels = 0
        self.cand_to_rel = {}
        self.start = time.time()
        self.summarization_flag = False
        if self.summarization is not None:
            self.summarization_flag = True
            summary_triples = self.summarization.summarize(pred[0], cands)
            if len(summary_triples) > 0:
                cands = summary_triples
            else:
                self.summarization_flag = False

        evaluated_cands = self.explore_singleton_cands(pred, cands)
        evaluated_cands = sort_by_relevance(evaluated_cands)


        triples_number = len(evaluated_cands)
        self.rels = triples_number

        best = evaluated_cands[0]["engine_output"]["rel"]

        self.cand_to_rel = {
            c["cand"][0]: c["engine_output"]["rel"] for c in evaluated_cands
        }

        if best <= self.xsi:
            length_cap = min(triples_number, self.length_cap) + 1
            for cand_length in range(2, length_cap):
                new_cands = self.explore_compound_cands(pred, cands, cand_length)
                evaluated_cands += new_cands
                best = max(cand["engine_output"]["rel"] for cand in new_cands)
                if best > self.xsi:
                    break

        evaluated_cands = sort_by_relevance_and_length(evaluated_cands)
        output_cands = self.process_output_cands(evaluated_cands)
        log_cands(i, output_cands)
        self.save_cands(i, output_cands)

        self.end = time.time()
        output = self.format_output(pred, output_cands)

        return output

    def explore_singleton_cands(self, pred, cands: list):
        evaluated_cands = []

        for cand in cands:
            mapped_cand = [cand]
            if self.summarization_flag:
                mapped_cand = self.summarization.map_rule(mapped_cand)

            engine_output = self.engine.compute_relevance(pred, mapped_cand)
            evaluated_cands.append({"cand": [cand], "engine_output": engine_output})
        return evaluated_cands

    def explore_compound_cands(self, pred, cands: list, length: int):
        compound_cands = itertools.combinations(cands, length)
        compound_cands = [(r, self.compute_cand_prescore(r)) for r in compound_cands]
        compound_cands = sorted(compound_cands, key=lambda x: x[1], reverse=True)

        terminate = False
        best = -1e6
        sliding_window = [None for _ in range(self.window_size)]

        evaluated_cands = []
        for i, (cand, _) in enumerate(compound_cands):
            if terminate:
                break
            mapped_cand = cand
            if self.summarization_flag:
                mapped_cand = self.summarization.map_rule(cand)

            engine_output = self.engine.compute_relevance(pred, mapped_cand)
            relevance = engine_output["rel"]
            self.rels += 1

            sliding_window[i % self.window_size] = relevance

            evaluated_cand = {"cand": cand, "engine_output": engine_output}

            if relevance > self.xsi:
                terminate = True
            elif relevance >= best:
                best = relevance
            elif i >= self.window_size:
                avg_window_relevance = sum(sliding_window) / self.window_size
                terminate_threshold = avg_window_relevance / best
                random_value = random.random()
                terminate = random_value > terminate_threshold

                evaluated_cand["random_value"] = random_value
                evaluated_cand["terminate_threshold"] = terminate_threshold
                evaluated_cand["terminate"] = terminate

            evaluated_cands.append(evaluated_cand)

        return evaluated_cands

    def compute_cand_prescore(self, cand):
        return sum([self.cand_to_rel[triple] for triple in cand])
