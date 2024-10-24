import os
import pandas as pd

import numpy

from pathlib import Path

from .utils import read_json

FS_METRICS_PATH = Path("fs_metrics")

def build_dataframe():
    files = os.listdir(FS_METRICS_PATH)

    results = []
    for file in files:
        if file == ".gitignore" or file == "README.md":
            continue
        fields = file.split('_')
        if fields[0] != "benchmark":
            experiment = {
                "method": fields[0],
                "mode": fields[1],
                "model": fields[2],
                "dataset": fields[3],
                "summarization": fields[4],
                "include-ranking": fields[5],
                "include-examples": fields[6],
                "llm": fields[7],
                "parameters": fields[8].split(".")[0],
            }

            metrics = read_json(FS_METRICS_PATH / file)

            experiment.update({
                "pre": float(metrics["avg_pre"]),
                "post": float(metrics["avg_post"]),
                "delta": float(metrics["delta"]), 
            })

            results.append(experiment)

    return results

def make_table():
    results = pd.DataFrame(build_dataframe())

    results = results.loc[results["llm"] == "Mixtral"]
    results = results.loc[results["parameters"] == "7B"]
    results = results.loc[results["include-ranking"] == "yes"]
    results = results.loc[results["include-examples"] == "yes"]
    results = results.loc[results["dataset"] != "WN18RR"]
    results = results.loc[results["dataset"] != "FB15k-237"]
    results = results.loc[results["dataset"] != "YAGO3-10"]

    results = results.replace("necessary", "nec")
    results = results.replace("sufficient", "suff")

    results = results.pivot_table(
        index=["model", "method", "mode", "summarization"],
        columns=["dataset"],
        values=["delta"],
        aggfunc=lambda x: x,
    )
    results = results.swaplevel(axis=1).sort_index(axis=1, level=0)

    latex = results.to_latex(na_rep="--", float_format="%.3f")
    latex = latex.replace("\multirow[t]{2}{*}{kelpie}", r"\textsc{Kelpie}")
    latex = latex.replace("\multirow[t]{2}{*}{criage}", r"\textsc{Criage}")
    latex = latex.replace("\multirow[t]{2}{*}{data-poisoning}", r"\textsc{DP}")
    latex = latex.replace("GEnI", r"\textsc{GEnI}")

    latex = latex.replace("TransE", r"\rotatebox[origin=c]{90}{\textsc{TransE}}")
    latex = latex.replace("ComplEx", r"\rotatebox[origin=c]{90}{\textsc{ComplEx}}")
    latex = latex.replace("ConvE", r"\rotatebox[origin=c]{90}{\textsc{ConvE}}")

    latex = latex.replace(r"\multirow[t]", r"\multirow[c]")
    latex = latex.replace(r"\multirow[c]{4}{*}{kelpie++}", r"\kelpiepp{}")
    latex = latex.replace(r"\multirow[c]{2}{*}{nec}", "nec")
    latex = latex.replace(r"\multirow[c]{2}{*}{suff}", "suff")
    latex = latex.replace("bisimulation", "b")
    latex = latex.replace("simulation", "s")
    latex = latex.replace("None", "--")

    latex = "\n".join([line for line in latex.split("\n") if "cline" not in line])
    latex = "\n".join(latex.split("\n")[6:-3])

    latex = "\n".join(latex.split("\n")[-9:]) + "\n" + "\n".join(latex.split("\n")[:-9])

    # add the string \midrule after 9 lines and then after 11 lines
    latex = "\n".join(latex.split("\n")[:9]) + "\n" + "\midrule\n" + "\n".join(latex.split("\n")[9:])
    latex = "\n".join(latex.split("\n")[:21]) + "\n" + "\midrule\n" + "\n".join(latex.split("\n")[21:])

    print(latex)

if __name__ == '__main__':
    make_table()
