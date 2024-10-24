import click

import wandb

import torch

from .. import DATASETS, MODELS
from .. import LP_CONFIGS_PATH, MODELS_PATH

from ..dataset import Dataset

from ..utils import read_json, write_json
from ..utils import init_model, init_optimizer
from ..utils import set_seeds


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--valid", type=float, default=-1, help="Number of epochs before valid.")
def main(dataset, model, valid):
    set_seeds(42)

    model_name = f"{model}_{dataset}"
    model_path = MODELS_PATH / f"{model_name}.pt"

    lp_config_path = LP_CONFIGS_PATH / f"{model}_{dataset}.json"
    lp_config = read_json(lp_config_path)    
    wandb.init(project="dixti", config=lp_config)
    wandb.log(
        {
            "execution_command": f"./scripts/lp_no_hpo {dataset} {model} {valid}"
        }
    )
    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Initializing model {model}...")
    model = init_model(lp_config, dataset)

    print("Training model...")
    optimizer = init_optimizer(lp_config, model)
    training_triples = dataset.training_triples
    validation_triples = dataset.validation_triples
    epochs = optimizer.train(training_triples, valid, validation_triples)

    print("\tSaving model...")
    torch.save(optimizer.model.state_dict(), model_path)

    wandb.log_model(model_path, name=model_name)

    lp_config["training"]["valid"] = valid
    lp_config["training"]["patience"] = 5
    lp_config["training"]["trained_epochs"] = epochs
    write_json(lp_config, lp_config_path)



if __name__ == "__main__":
    main()
