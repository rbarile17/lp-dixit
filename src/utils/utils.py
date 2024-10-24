import itertools
import random

import torch

import numpy as np

from .. import (
    CANDIDATES_PATH,
    CONFIGS_PATH,
    EVALUATIONS_PATH,
    EXPLANATIONS_PATH,
    METRICS_PATH,
    MODELS_PATH,
    RANKINGS_PATH,
    SAMPLED_SELECTED_PREDICTIONS_PATH,
)
from .. import IMAGINE
from .. import FIRST, NOT_FIRST

from ..link_prediction import MODEL_REGISTRY


pairs = lambda iterable: itertools.combinations(iterable, 2)


def jaccard_similarity(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())

def get_pred_rank(method):
    pred_rank = FIRST
    if method == IMAGINE:
        pred_rank = NOT_FIRST

    return pred_rank

def read_preds(method, model, dataset):
    pred_rank = get_pred_rank(method)
    preds = SAMPLED_SELECTED_PREDICTIONS_PATH / f"{model}_{dataset}_{pred_rank}.csv"
    with open(preds, "r") as preds:
        preds = [x.strip().split("\t") for x in preds.readlines()]

    return preds

def format_output_filename(method, mode, model, dataset, summarization):
    pred_rank = get_pred_rank(method)
    return f"{method}_{mode}_{model}_{dataset}_{summarization}_{pred_rank}"

def format_paths(method, mode, model, dataset, summarization):
    name = format_output_filename(method, mode, model, dataset, summarization)

    candidates = CANDIDATES_PATH / name
    candidates.mkdir(exist_ok=True)

    return {
        "configs": CONFIGS_PATH / f"{name}.json",
        "exps": EXPLANATIONS_PATH / f"{name}.json",
        "evals": EVALUATIONS_PATH / f"{name}.json",
        "metrics": METRICS_PATH / f"{name}.json",
        "candidates": CANDIDATES_PATH / name,
        "rankings": RANKINGS_PATH / f"{model}_{dataset}.json",
    }

def init_model(lp_config, dataset):
    model = lp_config["model"]
    model_class = MODEL_REGISTRY[model]["class"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**lp_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    return model

def load_model(lp_config, dataset):
    model = lp_config["model"]
    model_path = MODELS_PATH / f"{model}_{lp_config['dataset']}.pt"
    model = init_model(lp_config, dataset)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model

def init_optimizer(lp_config, model):
    hp = lp_config["training"]
    optimizer_class = MODEL_REGISTRY[lp_config["model"]]["optimizer"]
    optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
    optimizer = optimizer_class(model=model, hp=optimizer_params)

    return optimizer
