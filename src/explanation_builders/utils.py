import wandb

import pandas as pd


def sort_by_relevance(evaluated_cands):
    return sorted(evaluated_cands, key=lambda x: x["engine_output"]["rel"], reverse=True)

def sort_by_relevance_and_length(evaluated_cands):
    return sorted(
        evaluated_cands,
        key=lambda x: (x["engine_output"]["rel"], 1 / len(x["cand"])),
        reverse=True,
    )

def log_cands(i, cands):
    if cands != []:
        output_df = pd.DataFrame.from_records(cands)
        output_df["cand"] = output_df["cand"].apply(lambda e: [" ".join(t) for t in e])
        output_df["cand"] = output_df["cand"].map("\n".join)
        table = wandb.Table(dataframe=output_df)
        wandb.log({f"candidates_{i}": table})
