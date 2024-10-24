# LP-DIXIT

This repository contains the official code for the paper UNDER REVIEW that presents LP-DIXIT.
It evaluates explanations for _Link Prediction_ (LP) tasks on _Knowledge Graphs_ (KGs) in RDF.
It labels each explanation as **bad**, **neutral**, or **good**.
The label is the _Forward Simulatability Variation_ (FSV) which measures the variation of predictability of an inference (LP) caused by an explanations; an inference is _predictable_ if a verifier can hypothesize its output given the same input and without replicating the same process.
In the paper we focus on LP methods based on _Knowledge Graph Embeddings_ (KGE) and on _post-hoc_ methods to eXplain LP (LP-X).

LP-DIXIT employs a _Large Language Model_ (LLM) as a verifier to automate the evaluation!
It builds prompts by filling out our prompt template (sections separated by blank lines and variable parts in curly braces):

```
An RDF triple is a statement (subject, predicate, object). The subject and the object are entities, and the predicate is a relation between the subject and the object. Perform a Link Prediction (LP) task, specifically, given an incomplete RDF triple (subject, predicate, ?), predict the missing object that completes the triple and makes it a true statement.

Strict requirement: output solely the name of a single object entity, discard any explanation or other text.
Correct format: Italy
Incorrect format: The object entity is Italy.

({s}, {p}, ?)

{Explanation}
```

In the paper we propose four declinations of LP-DIXIT: LP-DIXIT, LP-DIXIT-O, LP-DIXIT-D, and LP-DIXIT-OD that combines the first two.
LP-DIXIT-O includes in the prompt a set of entities computed through the KGE model and a natural language instruction stating that the LLM must pick its response from such set, LP-DIXIT-D includes in the prompt a set of examples (or demonstrations) of solved LP queries, and LP-DIXIT-OD combines both LP-DIXIT-O, and LP-DIXIT-D.

Read our paper for a detailed formalization of the approach!

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

LP-DIXIT currently comes with the following datasets:

* FB15k-237
* WN18RR
* YAGO3-10
* DB50K
* DB100K
* YAGO4-20
* FRUNI
* FR200K

FRUNI and FR200K are benchamrk datasets that contain ground truth explanations for each prediction.

[Download the included datasets](https://figshare.com/s/cfe5eced5cf283afa016)

[Add other datasets](./data/README.md)

### KGE Models

LP-DIXIT currently provides an implementation for:

* TransE 
* ComplEx
* ConvE

We pre-trained such models on all the KGs.

[Download the pre-trained models](https://figshare.com/s/cfe5eced5cf283afa016)

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

We used this code to perform the experiments in our paper, specifically:

* we measured the alignment of LP-DIXIT with human judgement;
* we adopted LP-DIXIT to compare, on several well known KGs, state-of-the-art (SOTA) LP-X methods.

We used this code for all the phases of the experiments: from hyper-parameter tuning to evaluation of explanations.
Everything is reproducible, to start only the hyper-parameter ranges are needed!

The ground truth explanations in the benchmarks are in the explanations folder having `benchmark` as value for the field `method` in the file name.

[Download](https://figshare.com/s/cfe5eced5cf283afa016) experiment results, configurations, and pre/trained models. 

We executed the experiments on the cluster [Leonardo@Cineca](https://wiki.u-gov.it/confluence/display/SCAIUS/HPC+User+Guide) with Python 3.11.6

## Citation

TODO IF ACCEPTED