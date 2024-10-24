import click

import pandas as pd

from . import DATASETS, MODELS, PRED_RANKS
from . import PREDICTIONS_PATH, SELECTED_PREDICTIONS_PATH, SAMPLED_SELECTED_PREDICTIONS_PATH

from . import FIRST, NOT_FIRST

from .dataset import Dataset

from .utils import set_seeds

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
def main(dataset, model):
    set_seeds(42)

    preds_path = PREDICTIONS_PATH / f"{model}_{dataset}.csv"
    selected_preds_path = SELECTED_PREDICTIONS_PATH / f"{model}_{dataset}.csv"
    sampled_selected_preds_path = SAMPLED_SELECTED_PREDICTIONS_PATH / f"{model}_{dataset}.csv"

    dataset = Dataset(dataset)

    preds = pd.read_csv(preds_path, sep=";")
    preds.drop("s_rank", axis=1, inplace=True)

    preds = preds[preds["o_rank"] == 1]
    preds.drop(["o_rank"], axis=1, inplace=True)

    preds.to_csv(selected_preds_path, sep="\t", index=False, header=False)

    preds = preds.sample(100)
    preds = preds.reset_index(drop=True)

    preds.to_csv(sampled_selected_preds_path, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
