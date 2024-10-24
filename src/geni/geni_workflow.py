import operator

from .utils import get_clustered_elements
from .axiom_extraction import get_axioms, evaluate_fact_axiom
from .correlation_detection import (
    find_entity_correlations,
    find_direct_correlations,
    find_triangular_correlations,
    evaluate_direct_correlations,
    evaluate_triangular_correlations,
)
from .influence_detection import find_best_attack


def phase_1(
    pred,
    predicate_embs,
    predicate_clusters,
    th,
    model_type,
    facts,
    dataset
):
    s, p, o = pred
    search_space = get_clustered_elements(predicate_clusters, predicate_embs, p)
    axioms = get_axioms(th, predicate_embs, p, model_type, search_space)
    explanations = []
    if (
        axioms["symmetric"]
        or axioms["transitive"]
        or axioms["equivalent"]
        or axioms["inverse"]
        or axioms["chain"]
    ):
        trig_rules = evaluate_fact_axiom((s, o, p), axioms, facts, dataset)
        if trig_rules['symmetric']:
            t = dataset.labels_triple((o, p, s))
            explanation = f"The predicate {dataset.id_to_relation[p]} is symmetric. Hence, the RDF triple can be completed according to the known triple: {t}"
            explanations.append(explanation)
        if trig_rules['transitive']:
            transitive = f"The predicate {dataset.id_to_relation[p]} is transitive. Hence, the RDF triple can be completed according to the known triples: "
            triples = []
            for e in trig_rules['transitive']:
                t1 = (s, p, e)
                t2 = (e, p, o)
                t1 = dataset.labels_triple(t1)
                t2 = dataset.labels_triple(t2)
                triples.append(f"{t1}, {t2}")
            explanation = transitive + ", ".join(triples)
            explanations.append(explanation)
        if trig_rules["equivalent"]:
            for equivalent_predicate in trig_rules['equivalent']:
                t = (s, equivalent_predicate, o)
                t = dataset.labels_triple(t)
                ps = dataset.id_to_relation[p]
                eps = dataset.id_to_relation[equivalent_predicate]
                explanations.append(f"The predicate {ps} is equivalent to the predicate {eps}. Hence, the RDF triple can be completed according to the known triple: {t}")
        if trig_rules['inverse']:
            for inverse_predicate in trig_rules['inverse']:
                t = (o, inverse_predicate, s)
                t = dataset.labels_triple(t)
                ps = dataset.id_to_relation[p]
                ips = dataset.id_to_relation[inverse_predicate]

                explanations.append(f"The predicate {ps} is the inverse of the predicate {ips}. Hence, the RDF triple can be completed according to the known triple: {t}")
        if trig_rules['chain']:
            for chain in trig_rules['chain']:
                p1 = dataset.id_to_relation[chain["predicates"][0]]
                p2 = dataset.id_to_relation[chain["predicates"][1]]
                chain_s = f"The predicate {dataset.id_to_relation[p]} is the composition of the predicates {p1} and {p2}. Hence, the RDF triple can be completed according to the known triples:"
                triples = []
                for entity in chain["entities"]:
                    t1 = (s, chain["predicates"][0], entity)
                    t2 = (entity, chain["predicates"][1], o)

                    t1 = dataset.labels_triple(t1)
                    t2 = dataset.labels_triple(t2)

                    triples.append(f"{t1}, {t2}")
                explanations.append(chain_s + ", ".join(triples))

        return "\n".join(explanations)
    else:
        return ""


def phase_2(pred, ent_dict, rel_dict, e_clusters, known_facts, type, th_value, dataset):
    s, p, o = pred
    tail_corrs = find_entity_correlations(o, ent_dict, e_clusters, th_value)
    direct_correlations = find_direct_correlations(o, p, tail_corrs, rel_dict, ent_dict, type, th_value)
    if direct_correlations:
        eval_corrs = evaluate_direct_correlations(direct_correlations, known_facts)
        if 1 in list(eval_corrs.values()):
            corrs = [k for k, v in eval_corrs.items() if v == 1]
            corrs = sorted(corrs, key=operator.itemgetter(2), reverse=True)
            corr = corrs[0]
            t = (s, corr[1], corr[0])
            t = dataset.labels_triple(t)
            explanation = f"The RDF triple to complete is highly correlated with the known triple {t}"

            return explanation
        
    triangular_correlations = find_triangular_correlations(s, p, rel_dict, ent_dict, type, th_value)
    if triangular_correlations:
        corrs = evaluate_triangular_correlations(s, triangular_correlations, known_facts)
        if 1 in list(corrs.values()):
            corrs = [k for k, v in corrs.items() if v == 1]
            corrs = sorted(corrs, key=operator.itemgetter(0), reverse=True)
            e = corrs[0][0]
            explanation = f"The RDF triple to complete is highly correlated with the entity {e}"

            return explanation


    return ""


def phase_3(pred, ent_dict, rel_dict, known_facts, type, dataset):
    t = find_best_attack(pred, ent_dict, rel_dict, known_facts, type, dataset)
    t = dataset.labels_triples(t)
    t = [f"({s}, {p}, {o})" for s, p, o in t]

    if len(t) > 0:
        explanation = f"The RDF triple to complete is highly influenced by the known triples {','.join(t)}"
    else:
        explanation = ""

    return explanation
