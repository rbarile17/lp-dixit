# LP-DIXIT

[![DOI](https://zenodo.org/badge/794102618.svg)](https://doi.org/10.5281/zenodo.14875932)

This repository contains the official code for the paper "LP-DIXIT: Evaluating Explanations of Link Predictions on Knowledge Graphs using Large Language Models" that presents LP-DIXIT.
It evaluates explanations for _Link Prediction_ (LP) tasks on _Knowledge Graphs_ (KGs) in RDF.
It labels each explanation as **bad**, **neutral**, or **good**.
The label is the _Forward Simulatability Variation_ (FSV) which measures the variation of predictability of an inference (LP) caused by explanations; an inference is _predictable_ without (with) explanation if a verifier can hypothesize its output given the same input (and the explanation).
In the paper we focus on LP methods based on _Knowledge Graph Embeddings_ (KGE) and on _post-hoc_ methods to eXplain LP (LP-X).

LP-DIXIT employs a _Large Language Model_ (LLM) as a verifier to automate the evaluation!

Read the paper for a detailed formalization of the approach!

Check the Report on Additional Experiments!

## Installation

### Code

```bash
# Clone the repository
git clone

# Navigate to the repository directory
cd lp-dixit

# Install the required dependencies
pip install -r requirements.txt
```

### Data

LP-DIXIT supports any RDF KG.

[Read format details](./data/README.md)

### KGE Models

LP-DIXIT currently supports an implementation for:

* TransE 
* ComplEx
* ConvE

We pre-trained such models on all the KGs.

[Add other models](./kge_models/README.md)

## Usage

```python
python -m src.evaluation.llm_forward_simulatability [OPTIONS] FILE
```

It evaluates the explanations in `FILE`, which is a valid json file structured as follows:

```json
{
    "pred": ["<subject>", "<predicate>", "<object>"],
    "explanation": "<explanation>"
}
```

`<explanation>` is any string that can be framed in a prompt, for instance, it can be a set of RDF triples.

### Options:

`--dataset <dataset>`

`--model <model>`

`--llm <llm>` Use the LLM `<llm>` as a verifier, `<llm>` is Llama-3.1 or Mixtral  

`--parameters <N>` Use the specified LLM in the version with `<N>` billions of parameters, `<N>` is 8 or 70 when `<llm>` is Llama-3.1, while `<N>` is 7 or 22 when `<llm>` is Mixtral

`--include-ranking` Execute LP-DIXIT-O instead of plain LP-DIXIT, this flag can also be combined with `--include-examples`

`--include-examples` Execute LP-DIXIT-D instead of plain LP-DIXIT, this flag can also be combined with `--include-ranking`

`--help`

### Output:

This command writes two files:

* `fs_details/{FILE}_{include-ranking}_{include-examples}_{llm}_{parameters}B.json`: label and intermediate results for each explanation
* `fs_metrics/{FILE}_{include-ranking}_{include-examples}_{llm}_{parameters}B.json`: aggregate metrics

Note that the command removes `.json` from `FILE` when creating the output files.

## Additional Features

### Tune hyper-parameters of a KGE model

```python
python -m src.link_prediction.tune --dataset <dataset> --model <model>
```

It saves the best hyper-parameters found in `lp_config/{model}_{dataset}.json`.

### Train a KGE model

```python
python -m src.link_prediction.train --dataset <dataset> --model <model> --valid <validation_epochs>
```

`<valid>` is the frequency (in epochs) of evaluation of the model on the validation set to determine whether to apply early stopping

The command reads the model configuration specified in `lp_config/{model}_{dataset}.json`, generate it through hyper-parameters tuning or create it manually.

It saves the model parameters in `kge_models/{model}_{dataset}.pt`.

### Make predictions (compute the rank of test triples)

```python
python -m src.link_prediction.test --dataset <dataset> --model <model>
```

It reads the model parameters from `kge_models/{model}_{dataset}.pt` and saves the predictions in `preds/{model}_{dataset}.csv`.

### Select and sample 100 correct predictions (top-ranked test triples)

```python
python -m src.select_preds --dataset <dataset> --model <model>
```

It reads the predictions from `preds/{model}_{dataset}.csv` and saves the correct ones in `selected_preds/{model}_{dataset}`; then it saves a sample of the correct prediction in `sampled_selected_preds/{model}_{dataset}`.

### Generate explanations

```python
python -m src.explain [OPTIONS]
```
#### Options

`--dataset <dataset>`

`--model <model>`

`--method <method>` Use the LP-X method `<method>`, it is **criage**, **data-poisoning**, **kelpie**, **kelpie++**, **GEnI**;

`--mode <mode>` Run `<method>` in mode `<mode>`, it is **necessary**, or **sufficient**; this parameter is ignored by **GEnI** as it runs in a single mode.

`--summarization <summarization>` Execute the method with `<summarization>` as _Graph Summarization_ strategy, `<summarization>` is **simulation**, or **bisimulation**; this parameter is needed only when `<method>` is kelpie++.

#### Input and Output

It explains the predictions in `sampled_selected_preds/{model}_{dataset}`; hence, it saves the explanations in `explanations/{method}_{mode}_{model}_{dataset}_{summarization}.json`.

Generate the predictions with the previous commands or upload your own predictions!

## Experiments

We used this code to perform all the experiments in the paper.

[Download KGs](https://doi.org/10.6084/m9.figshare.27292017.v1)

[Download learned embeddings](https://doi.org/10.6084/m9.figshare.28424246) 

[Download experiment results and execution logs](https://doi.org/10.6084/m9.figshare.28424321)

We executed the experiments on the cluster [Leonardo@Cineca](https://wiki.u-gov.it/confluence/display/SCAIUS/HPC+User+Guide) with Python 3.11.6

## ACM Reference

Roberto Barile, Claudia d’Amato, and Nicola Fanizzi. 2025. LP-DIXIT: Evaluating Explanations of Link Predictions on Knowledge Graphs using Large Language Models. In Proceedings of the ACM Web Conference 2025 (WWW ’25), April 28–May 2, 2025, Sydney, NSW, Australia. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3696410.3714667
