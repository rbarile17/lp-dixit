import wandb

import click

import pandas as pd

from tqdm import tqdm

from . import DATASETS, METHODS, MODELS, MODES
from . import LP_CONFIGS_PATH
from . import CRIAGE, DATA_POISONING, IMAGINE, KELPIE, KELPIEPP, GENI
from . import NECESSARY, SUFFICIENT

from .dataset import Dataset
from .explanation_builders import CriageBuilder, DataPoisoningBuilder, StochasticBuilder
from .explanation_builders.summarization import SUMMARIZATIONS
from .pipeline import ImaginePipeline, NecessaryPipeline, SufficientPipeline, GENIPipeline
from .prefilters import TopologyPreFilter, CriagePreFilter
from .relevance_engines import (
    NecessaryCriageEngine,
    SufficientCriageEngine,
    NecessaryDPEngine,
    SufficientDPEngine,
    ImaginePostTrainingEngine,
    NecessaryPostTrainingEngine,
    SufficientPostTrainingEngine
)

from .utils import read_json, write_json
from .utils import read_preds
from .utils import format_paths
from .utils import load_model
from .utils import set_seeds

def build_pipeline(model, dataset, hp, method, mode, summarization, candidates_path, model_name):
    if method in [CRIAGE, DATA_POISONING, KELPIE, KELPIEPP]:
        if mode == NECESSARY:
            if method == CRIAGE:
                prefilter = CriagePreFilter(dataset)
                engine = NecessaryCriageEngine(model, dataset)
                builder = CriageBuilder(engine)
            elif method == DATA_POISONING:
                prefilter = TopologyPreFilter(dataset)
                engine = NecessaryDPEngine(model, dataset, hp["lr"])
                builder = DataPoisoningBuilder(engine)
            elif method == KELPIE or method == KELPIEPP:
                DEFAULT_XSI_THRESHOLD = 5
                xsi = DEFAULT_XSI_THRESHOLD
                prefilter = TopologyPreFilter(dataset)
                engine = NecessaryPostTrainingEngine(model, dataset, hp)
                builder = StochasticBuilder(xsi, engine, summarization=summarization)
            builder.candidates_path = candidates_path
            pipeline = NecessaryPipeline(dataset, prefilter, builder)
        elif mode == SUFFICIENT:
            criage = False
            if method == CRIAGE:
                prefilter = CriagePreFilter(dataset)
                engine = SufficientCriageEngine(model, dataset)
                builder = CriageBuilder(engine)
                criage = True
            elif method == DATA_POISONING:
                prefilter = TopologyPreFilter(dataset)
                engine = SufficientDPEngine(model, dataset, hp["lr"])
                builder = DataPoisoningBuilder(engine)
            elif method == KELPIE or method == KELPIEPP:
                DEFAULT_XSI_THRESHOLD = 0.9
                xsi = DEFAULT_XSI_THRESHOLD
                prefilter = TopologyPreFilter(dataset)
                engine = SufficientPostTrainingEngine(model, dataset, hp)
                builder = StochasticBuilder(xsi, engine, summarization=summarization)
            builder.candidates_path = candidates_path
            pipeline = SufficientPipeline(dataset, prefilter, builder, criage=criage)
    elif method == IMAGINE:
        DEFAULT_XSI_THRESHOLD = 0.9
        xsi = DEFAULT_XSI_THRESHOLD
        prefilter = TopologyPreFilter(dataset)
        engine = ImaginePostTrainingEngine(model, dataset, hp)
        builder = StochasticBuilder(xsi, engine, summarization=summarization)
        builder.candidates_path = candidates_path
        pipeline = ImaginePipeline(dataset, prefilter, builder)
    elif method == GENI:
        pipeline = GENIPipeline(dataset, model, model_name, 0.6)

    return pipeline


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS), default=IMAGINE)
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

    model_name = model

    paths = format_paths(method, mode, model, dataset, summarization)

    lp_config_path = LP_CONFIGS_PATH / f"{model}_{dataset}.json"
    lp_config = read_json(lp_config_path)
    lp_config["training"]["epochs"] = lp_config["training"]["trained_epochs"]

    config = {
        "method": method,
        "mode": mode,
        "summarization": summarization,
    }

    wandb.init(project="dixti", config=config)
    wandb.log(
        {
            "execution_command": f"./scripts/explain {method} {mode} {dataset} {model} {summarization}"
        }
    )

    print("Reading preds...")
    preds = read_preds(method, model, dataset)

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model = load_model(lp_config, dataset)
    model.eval()

    pipeline_hps = lp_config["training"]
    pipeline = build_pipeline(
        model,
        dataset,
        pipeline_hps,
        method,
        mode,
        summarization,
        paths["candidates"],
        model_name
    )

    write_json(config, paths["configs"])

    explanations = []
    i = 0
    for pred in tqdm(preds):
        pred = dataset.ids_triple(pred)
        explanation = pipeline.explain(i, pred)
        explanations.append(explanation)
        i += 1
    write_json(explanations, paths["exps"])

    # exp_df = pd.DataFrame.from_records(explanations)
    # exp_df["pred"] = exp_df["pred"].map(" ".join)
    # exp_df["explanation"] = exp_df["explanation"].apply(lambda e: [" ".join(t) for t in e])
    # exp_df["explanation"] = exp_df["explanation"].map("\n".join)
    # if mode == SUFFICIENT:
    #     exp_df["entities_to_convert"] = exp_df["entities_to_convert"].map(" ".join)
    # table = wandb.Table(dataframe=exp_df)
    # wandb.log({"explanations": table})


if __name__ == "__main__":
    main()
