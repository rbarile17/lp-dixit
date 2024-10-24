import wandb

import copy
import click

import pandas as pd

import numpy as np

from collections import defaultdict

from .. import DATASETS, METHODS, MODELS, MODES
from .. import CRIAGE, DATA_POISONING, IMAGINE, KELPIE, KELPIEPP
from .. import NECESSARY, SUFFICIENT

from .. import LP_CONFIGS_PATH

from ..dataset import MANY_TO_ONE, ONE_TO_ONE

from ..explanation_builders.summarization import SUMMARIZATIONS

from ..dataset import Dataset
from ..link_prediction.evaluation import Evaluator

from ..utils import format_paths
from ..utils import read_json, write_json
from ..utils import init_model, load_model
from ..utils import init_optimizer
from ..utils import set_seeds

def format_result(results, new_results, pred):
    result = results[pred]
    new_result = new_results[pred]

    score = result["score"]["tail"]
    rank = result["rank"]["tail"]
    new_score = new_result["score"]["tail"]
    new_rank = new_result["rank"]["tail"]

    return {
        "score": str(score),
        "rank": str(rank),
        "new_score": str(new_score),
        "new_rank": str(new_rank),
    }

def get_results(model, triples):
    evaluator = Evaluator(model=model)
    evaluator.evaluate(np.array(triples))
    results = evaluator.results
    results = {c: result for c, result in zip(triples, results)}

    return results

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS), default=KELPIE)
@click.option("--mode", type=click.Choice(MODES))
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS))
def main(
    dataset,
    model,
    method,
    mode,
    summarization,
):
    set_seeds(42)

    paths = format_paths(method, mode, model, dataset, summarization)

    lp_config_path = LP_CONFIGS_PATH / f"{model}_{dataset}.json"
    lp_config = read_json(lp_config_path)
    lp_config["training"]["epochs"] = lp_config["training"]["trained_epochs"]
    config = read_json(paths["configs"])
    outputs = read_json(paths["exps"])

    wandb.init(project="dixti", config=config)

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model = load_model(lp_config, dataset)
    model.eval()

    preds = []

    if method in [CRIAGE, DATA_POISONING, KELPIE, KELPIEPP]:
        if mode == SUFFICIENT:
            conversions = []
            additions = []
            conversions_to_additions = defaultdict(list)
            pred_to_conversions = defaultdict(list)
            pred_to_explanation = defaultdict(list)
            for output in outputs:
                pred = dataset.ids_triple(output["pred"])
                s, p, _ = pred
                preds.append(pred)

                entities = output["entities_to_convert"]
                entities = [dataset.entity_to_id[entity] for entity in entities]

                explanation = output["explanation"]
                explanation = dataset.ids_triples(explanation)

                pred_to_explanation[pred] = explanation

                for entity in entities:
                    conversion = Dataset.replace_entity_in_triple(pred, s, entity)
                    additions_ = Dataset.replace_entity_in_triples(explanation, s, entity)
                    conversions_to_additions[conversion] = additions_
                    conversions.append(conversion)
                    pred_to_conversions[pred].append(conversion)
                    additions.extend(additions_)

            new_dataset = copy.deepcopy(dataset)

            for s, p, _ in additions:
                if new_dataset.relation_to_type[p] in [MANY_TO_ONE, ONE_TO_ONE]:
                    for existing_o in new_dataset.train_to_filter[(s, p)]:
                        new_dataset.remove_training_triple((s, p, existing_o))

            new_dataset.add_training_triples(additions)

            results = get_results(model, conversions)

            new_model = init_model(lp_config, new_dataset)
            optimizer = init_optimizer(lp_config, new_model)

            optimizer.train(training_triples=new_dataset.training_triples)

            new_model.eval()
            new_results = get_results(new_model, conversions)

            evaluations = []
            for pred in preds:
                conversions = pred_to_conversions[pred]
                explanation = pred_to_explanation[pred]
                explanation = dataset.labels_triples(explanation)
                evaluation = {"pred": dataset.labels_triple(pred), "explanation": explanation}
                for i, conv in enumerate(conversions):
                    result = format_result(results, new_results, conv)
                    additions = conversions_to_additions[conv]
                    additions = dataset.labels_triples(additions)
                    evaluation[f"conversion_{i}.additions"] = additions
                    evaluation[f"conversion_{i}.result.score"] = result["score"]
                    evaluation[f"conversion_{i}.result.rank"] = result["rank"]
                    evaluation[f"conversion_{i}.result.new_score"] = result["new_score"]
                    evaluation[f"conversion_{i}.result.new_rank"] = result["new_rank"]
                evaluations.append(evaluation)
        if mode == NECESSARY:
            removals = []
            pred_to_explanation = defaultdict(list)
            for output in outputs:
                pred = dataset.ids_triple(output["pred"])
                preds.append(pred)
                explanation = output["explanation"]
                explanation = [dataset.ids_triple(triple) for triple in explanation]
                pred_to_explanation[pred] = explanation
                removals += explanation

            new_dataset = copy.deepcopy(dataset)

            new_dataset.remove_training_triples(removals)

            results = get_results(model, preds)

            new_model = init_model(lp_config, new_dataset)
            optimizer = init_optimizer(lp_config, new_model)

            optimizer.train(training_triples=new_dataset.training_triples)

            new_model.eval()
            new_results = get_results(new_model, preds)

            evaluations = []
            for pred in preds:
                result = format_result(results, new_results, pred)

                explanation = pred_to_explanation[pred]
                explanation = [dataset.labels_triple(triple) for triple in explanation]

                evaluation = {
                    "pred": dataset.labels_triple(pred),
                    "explanation": explanation,
                    "result": result
                }

                evaluations.append(evaluation)
    elif method == IMAGINE:
        additions = []
        pred_to_explanation = defaultdict(list)
        for output in outputs:
            pred = dataset.ids_triple(output["pred"])
            preds.append(pred)
            explanation = output["explanation"]
            explanation = [dataset.ids_triple(triple) for triple in explanation]

            pred_to_explanation[pred] = explanation
            additions += explanation

        new_dataset = copy.deepcopy(dataset)

        new_dataset.add_training_triples(additions)

        results = get_results(model, preds)

        new_model = init_model(lp_config, new_dataset)
        optimizer = init_optimizer(lp_config, new_model)

        optimizer.train(training_triples=new_dataset.training_triples)

        new_model.eval()
        new_results = get_results(new_model, preds)

        evaluations = []
        for pred in preds:
            result = format_result(results, new_results, pred)

            explanation = pred_to_explanation[pred]
            explanation = [dataset.labels_triple(triple) for triple in explanation]
            evaluation = {
                "pred": dataset.labels_triple(pred),
                "explanation": explanation,
                "result": result
            }

            evaluations.append(evaluation)

    write_json(evaluations, paths["evals"])
    eval_df = pd.DataFrame.from_records(evaluations)
    eval_df["pred"] = eval_df["pred"].map(" ".join)
    if mode == SUFFICIENT:
        for i in range(10):
            eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].fillna("")
            eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].map(lambda e: [] if e == "" else e)
            eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].map(lambda e: [" ".join(t) for t in e])
            eval_df[f"conversion_{i}.additions"] = eval_df[f"conversion_{i}.additions"].map("\n".join)
    eval_df["explanation"] = eval_df["explanation"].apply(lambda e: [" ".join(t) for t in e])
    eval_df["explanation"] = eval_df["explanation"].map("\n".join)
    table = wandb.Table(dataframe=eval_df)
    wandb.log({"evaluations": table})

if __name__ == "__main__":
    main()
