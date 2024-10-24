import numpy as np

from .relevance_engines import PostTrainingEngine
from .fact_builder import FactBuilder
from .geni import phase_1, phase_2, phase_3
from .geni import utils


class Pipeline:
    def __init__(self, dataset, prefilter, builder):
        self.dataset = dataset
        self.prefilter = prefilter
        self.builder = builder

        self.engine = self.builder.engine
        self.model = self.engine.model

    def set(self):
        if isinstance(self.engine, PostTrainingEngine):
            self.engine.set_cache()


class ImaginePipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder):
        super().__init__(dataset, prefilter, builder)
        self.fact_builder = FactBuilder(dataset)

    def explain(self, i, pred, prefilter_k=20):
        super().set()

        s, _, _ = pred
        new_triples = self.fact_builder.build_triples(s, pred)

        triples = self.prefilter.select_triples(
            pred=pred, k=prefilter_k, new_triples=new_triples
        )
        result = self.builder.build_explanations(i, pred, triples)

        return result


class NecessaryPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder):
        super().__init__(dataset, prefilter, builder)

    def fail(self, pred):
        return {
            "pred": self.dataset.labels_triple(pred),
            "explanation": [],
            "#relevances": 0,
            "execution_time": 0,
        }   

    def explain(self, i, pred, prefilter_k=20):
        super().set()
        filtered_triples = self.prefilter.select_triples(pred=pred, k=prefilter_k)
        if filtered_triples == []:
            return self.fail(pred)

        result = self.builder.build_explanations(i, pred, filtered_triples)

        return result


class SufficientPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder, criage):
        super().__init__(dataset, prefilter, builder)
        self.criage = criage

    def fail(self, pred, entities_to_convert=[]):
        return {
            "pred": self.dataset.labels_triple(pred),
            "explanation": [],
            "#relevances": 0,
            "execution_time": 0,
            "entities_to_convert": entities_to_convert,
        }

    def explain(self, i, pred, prefilter_k=20, to_convert_k=10):
        self.engine.select_entities_to_convert(
            pred, to_convert_k, 200, criage=self.criage
        )
        if self.engine.entities_to_convert == []:
            return self.fail(pred)

        super().set()
        entities_to_conv = self.engine.entities_to_convert
        entities_to_conv = [self.dataset.id_to_entity[x] for x in entities_to_conv]
        filtered_triples = self.prefilter.select_triples(pred=pred, k=prefilter_k)
        if filtered_triples == []:
            return self.fail(pred, entities_to_conv)

        result = self.builder.build_explanations(i, pred, filtered_triples)

        result["entities_to_convert"] = entities_to_conv

        return result


class GENIPipeline:
    def __init__(self, dataset, model, model_name, user_th):
        self.dataset = dataset

        self.model_type = "semantic" if model_name == "ComplEx" else "translation"

        self.entity_embs = {
            e: np.squeeze(model.entity_embeddings[e].cpu().detach().numpy())
            for e in dataset.entity_to_id.values()
        }

        self.predicate_embs = {
            p: np.squeeze(model.relation_embeddings[p].cpu().detach().numpy())
            for p in dataset.relation_to_id.values()
        }

        print('Generating clusters')
        max_k = int(len(self.predicate_embs) / 2)
        self.predicate_clusters = utils.get_optimal_clusters(max_k, self.predicate_embs, self.model_type)
        self.entity_clusters = utils.get_entity_clusters(self.entity_embs)

        max = np.max(np.array(list(self.entity_embs.values())))
        min = np.min(np.array(list(self.entity_embs.values())))
        th = abs(max - min)
        self.th = th + (1 - user_th) * th

    def explain(self, i, pred):
        explanation = phase_1(
            pred,
            self.predicate_embs,
            self.predicate_clusters,
            self.th,
            self.model_type,
            self.dataset.predicate_to_training_triples,
            self.dataset
        )

        if explanation == "":
            explanation = phase_2(
                pred,
                self.entity_embs,
                self.predicate_embs,
                self.entity_clusters,
                self.dataset.predicate_to_training_triples,
                self.model_type,
                self.th,
                self.dataset
            )

            if explanation == "":
                explanation = phase_3(
                    pred,
                    self.entity_embs,
                    self.predicate_embs,
                    self.dataset.predicate_to_training_triples,
                    self.model_type,
                    self.dataset
                )

                return {
                    "pred": self.dataset.labels_triple(pred),
                    "explanation": explanation,
                }   
            else:
                return {
                    "pred": self.dataset.labels_triple(pred),
                    "explanation": explanation,
                }
        else:
            return {
                "pred": self.dataset.labels_triple(pred),
                "explanation": explanation,
            }
