import pandas as pd

from . import METHODS, MODES, DATASETS, MODELS
from .explanation_builders.summarization import SUMMARIZATIONS

DATASETS = [d for d in DATASETS if d != 'FR200K']
DATASETS = [d for d in DATASETS if d != 'FRUNI']
DATASETS = [d for d in DATASETS if d != 'FTREE']

METHODS = [m for m in METHODS if m != 'GEnI']
METHODS = [m for m in METHODS if m != 'imagine']

SUMMARIZATIONS = [s for s in SUMMARIZATIONS if s != 'no']

experiments = []
for method in METHODS:
    criage = False
    kelpiepp = False
    if method == 'criage':
        criage = True
    if method == "kelpie++":
        kelpiepp = True
    for mode in MODES:
        for dataset in DATASETS:
            for model in MODELS:
                if criage and model == "TransE":
                    continue
                for i in range(2):
                    for j in range(2):
                        if kelpiepp and dataset in ['DB50K', 'DB100K', 'YAGO4-20']:
                            for summarization in SUMMARIZATIONS:
                                experiments.append({
                                    'method': method,
                                    'mode': mode,
                                    'dataset': dataset,
                                    'model': model,
                                    'summarization': summarization,
                                    'include_ranking': i,
                                    'include_examlpes': j
                                })
                        else:
                            if not kelpiepp:
                                experiments.append({
                                    'method': method,
                                    'mode': mode,
                                    'dataset': dataset,
                                    'model': model,
                                    'summarization': "no",
                                    'include_ranking': i,
                                    'include_examlpes': j
                                })



df = pd.DataFrame(experiments)

df.index = df.index + 1
df.to_csv('output.txt', sep=' ')
