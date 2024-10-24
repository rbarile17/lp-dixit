import click

import torch

import numpy as np

from tqdm import tqdm

from .. import DATASETS, MODELS
from .. import LP_CONFIGS_PATH
from .. import RANKINGS_PATH

from ..dataset import Dataset

from ..utils import load_model
from ..utils import read_preds
from ..utils import read_json, write_json
from ..utils import set_seeds


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
def main(dataset, model):
    set_seeds(42)

    lp_config_path = LP_CONFIGS_PATH / f"{model}_{dataset}.json"
    lp_config = read_json(lp_config_path)

    out_path = RANKINGS_PATH / f"{model}_{dataset}.json"

    print("Reading preds...")
    preds = read_preds("kelpie", model, dataset)

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Initializing model {model}...")
    model = load_model(lp_config, dataset)
    model.eval()

    triples = dataset.ids_triples(preds)
    rankings = []
    for triple in tqdm(triples):
        all_scores = model.all_scores(np.expand_dims(triple, axis=0))[0]
        all_scores = torch.argsort(all_scores, descending=(not model.is_minimizer()))

        labels_triple = dataset.labels_triple(triple)
        ranking = [dataset.id_to_entity[j.item()] for j in all_scores]
        rankings.append({"pred": labels_triple, "ranking": ranking})

    write_json(rankings, out_path)


if __name__ == "__main__":
    main()
