import click

import wandb

from .evaluation import Evaluator

from .. import DATASETS, MODELS
from .. import LP_CONFIGS_PATH, PREDICTIONS_PATH

from ..dataset import Dataset

from ..utils import load_model
from ..utils import read_json
from ..utils import set_seeds


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
def main(dataset, model):
    set_seeds(42)

    preds_path = PREDICTIONS_PATH / f"{model}_{dataset}.csv"

    lp_config_path = LP_CONFIGS_PATH / f"{model}_{dataset}.json"
    lp_config = read_json(lp_config_path)

    wandb.init(project="dixti", config=lp_config)

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Initializing model {model}...")
    model = load_model(lp_config, dataset)
    model.eval()

    print("Evaluating model...")
    evaluator = Evaluator(model=model)
    ranks = evaluator.evaluate(triples=dataset.testing_triples)
    metrics = evaluator.get_metrics(ranks)
    wandb.log(metrics)
    df_output = evaluator.get_df_output(triples=dataset.testing_triples, ranks=ranks)

    # entity_dict = {k: model.entity_embeddings[v].cpu().detach().numpy() for k, v in dataset.entity_to_id.items()}

    df_output.to_csv(preds_path, index=False, sep=";")
    artifact = wandb.Artifact('preds', type='predictions')
    artifact.add_file(preds_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()
