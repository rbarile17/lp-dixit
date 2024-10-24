import click

from .. import DATASETS, METHODS, MODELS, MODES
from .. import IMAGINE, KELPIE
from .. import NECESSARY, SUFFICIENT

from ..explanation_builders.summarization import SUMMARIZATIONS

from ..utils import format_paths
from ..utils import read_json, write_json


def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return round(count / float(len(ranks)), 3)


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0 / float(rank)
    return round(reciprocal_rank_sum / float(len(ranks)), 3)


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return round(rank_sum / float(len(ranks)), 3)


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS), default=KELPIE)
@click.option("--mode", type=click.Choice(MODES))
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS))
def main(dataset, model, method, mode, summarization):
    paths = format_paths(method, mode, model, dataset, summarization)
    evals = read_json(paths["evals"])

    if mode == NECESSARY or method == IMAGINE:
        ranks = [float(pred["result"]["rank"]) for pred in evals]
        new_ranks = [float(pred["result"]["new_rank"]) for pred in evals]
    elif mode == SUFFICIENT:
        ranks = [float(pred.get(f"conversion_{i}.result.rank", -1)) for pred in evals for i in range(10)]
        new_ranks = [float(pred.get(f"conversion_{i}.result.new_rank", -1)) for pred in evals for i in range(10)]
        ranks = [x for x in ranks if x != -1]
        new_ranks = [x for x in new_ranks if x != -1]


    original_mrr, original_h1 = mrr(ranks), hits_at_k(ranks, 1)
    new_mrr, new_h1 = mrr(new_ranks), hits_at_k(new_ranks, 1)
    mrr_delta = round(new_mrr - original_mrr, 3)
    h1_delta = round(new_h1 - original_h1, 3)

    exps = read_json(paths["exps"])
    rels = [x["#relevances"] for x in exps]
    times = [x["execution_time"] for x in exps]
    rels = sum(rels)
    time = sum(times)

    metrics = {}
    metrics["delta_h1"] = h1_delta
    metrics["delta_mrr"] = mrr_delta
    metrics["rels"] = rels
    metrics["time"] = time

    write_json(metrics, paths["metrics"])

if __name__ == "__main__":
    main()
