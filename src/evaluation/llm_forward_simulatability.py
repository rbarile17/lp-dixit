import click
import random

import time
import torch

import pandas as pd

from ast import literal_eval

from pathlib import Path

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.example_selectors.base import BaseExampleSelector

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline

from ..dataset import Dataset

from ..link_prediction.evaluation import Evaluator

from ..utils import load_model

from ..utils import format_paths
from ..utils import read_json, write_json

from ..utils import set_seeds

from ..explanation_builders.summarization import SUMMARIZATIONS

from .. import DATASETS, MODELS, METHODS, MODES
from .. import DATA_PATH, LP_CONFIGS_PATH

from .. import DB50K, DB100K, YAGO4_20

FS_CONFIGS_PATH = Path("fs_configs")
FS_METRICS_PATH = Path("fs_metrics")
FS_DETAILS_PATH = Path("fs_details")

prefix_llama = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant.
Your response should be crisp, short and not repetitive.
Discard any preamble, explanation, greeting, or final consideration.
An RDF triple is a statement (subject, predicate, object).
The subject and the object are entities, and the predicate is a relation between the subject and the object.
Perform a Link Prediction (LP) task, specifically, given an incomplete RDF triple (subject, predicate, ?), predict the missing object that completes the triple and makes it a true statement.
Strict requirement: output solely the name of a single object entity, discard any explanation or other text. 
Correct format: Elizabeth_of_Bohemia
Incorrect format: The object entity is Elizabeth_of_Bohemia.
{ranking}
"""

suffix_llama = """
<|eot_id|><|start_header_id|>user<|end_header_id|>
({subject}, {predicate}, ?)
{explanation}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

example_prompt_str_template_llama = """
<|eot_id|><|start_header_id|>user<|end_header_id|>
({subject}, {predicate}, ?)
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{object}
"""

prefix_mistral = """
<s> [INST]
You are a helpful, respectful and honest assistant.
Your response should be crisp, short and not repetitive.
Discard any preamble, explanation, greeting, or final consideration.
An RDF triple is a statement (subject, predicate, object).
The subject and the object are entities, and the predicate is a relation between the subject and the object.
Perform a Link Prediction (LP) task, specifically, given an incomplete RDF triple (subject, predicate, ?), predict the missing object that completes the triple and makes it a true statement.
Strict requirement: output solely the name of a single object entity, discard any explanation or other text. 
Correct format: Elizabeth_of_Bohemia
Incorrect format: The object entity is Elizabeth_of_Bohemia.
{ranking}
[/INST]
"""

suffix_mistral = """
[INST]
({subject}, {predicate}, ?)
{explanation}
[/INST]
</s>
"""

example_prompt_str_template_mistral = """
[INST] ({subject}, {predicate}, ?) [/INST] <s> {object} </s>
"""

explanation_hook = """
In addition to the incomplete triple, an explanation is provided.
An explanation is a set of RDF triples that provide context for the prediction.
Given an incomplete triple (s, p, ?), all the RDF triples in the explanation feature s as subject or as object and connect it with another entity.
Explanation:
"""

qhook = """
This explanation is also provided in the form of quotient triples to provide a more abstract view of the explanation.
A quotient triple is an RDF triple featuring the subject entity of the incomplete triple as subject or as object and connects it with a concept name rather than an entity.
Considering an example explanation with the following triples:
(s, p, e)
(s, p, f)
(s, p, g)

The corresponding quotient triples are:
(s, p, C1)
(s, p, C2)

where C1, C2 are concept names, e and f are instances of C1, and g is an instance of C2.

Your answer still must be a single entity name even if the quotient triples contain concept names.

Finally, the quotient triples are:
"""

ranking_hook = """
The entity name that you provide must be in the following list of entity names:
"""

class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, dataset, df_output): 
        self.dataset = dataset

        self.df_output = df_output

    def add_example(example):
        return

    def select_examples(self, input_variables):
        predicate = input_variables["predicate"]

        examples = self.df_output[self.df_output["p"] == predicate].copy()
        examples = examples[examples["o_rank"] == 1]
        examples = examples.drop(columns=["s_rank", "o_rank"])
        examples = examples.rename(columns={"s": "subject", "p": "predicate", "o": "object"})

        examples = examples.to_dict("records")

        return random.sample(examples, 10) if len(examples) > 10 else examples
    
def build_explanation(explained_pred, summarization, e_sem):
    if summarization is None:
        explanation = explained_pred["explanation"]
        explanation = [f"({s}, {p}, {o})\n" for (s, p, o) in explanation]
        explanation = "\n".join(explanation)
        explanation = f"{explanation_hook}\n{explanation}"

        return explanation
    elif summarization == "simulation" or summarization == "bisimulation":
        triples = explained_pred["explanation"]
        triples_string = [f"({s}, {p}, {o})" for (s, p, o) in triples]
        triples_string = "\n".join(triples_string)

        pred_s = explained_pred["pred"][0]

        qtriples = explained_pred["quotient_explanation"]
        qtriples_string = []
        types = ""
        if qtriples != None:
            for qtriple in qtriples:
                s, p, o = qtriple
                if len(s) == 1 and s[0] == pred_s:
                    s_ = s[0]
                else:
                    if len(e_sem[s[0]]) == 0:
                        e_sem[s[0]] = "Thing"
                    s_ = f"[{e_sem[s[0]]}]"
                if len(o) == 1 and o[0] == pred_s:
                    o_ = o[0]
                else:
                    if len(e_sem[o[0]]) == 0:
                        e_sem[o[0]] = "Thing"
                    o_ = f"[{e_sem[o[0]]}]"

                if s_ == o_:
                    s_ = pred_s
                qtriples_string.append(f"({s_}, {p}, {o_})")
            
            types = [f"{s} is an instance of {e_sem[s]}" for s, _, _ in triples if s != pred_s]
            types += [f"{o} is an instance of {e_sem[o]}" for _, _, o in triples if o != pred_s]

            types = "\n".join(types)

        qtriples_string = "\n".join(qtriples_string)
        explanation = f"{explanation_hook}\n{triples_string}\n{qhook}\n{qtriples_string}\n{types}"

        return explanation

def build_ranking(ranking):
    ranking = [o for o in ranking]
    ranking = "\n".join(ranking)
    ranking = f"{ranking_hook}{ranking}"

    return ranking

def parse(response, llm):
    if llm == "Llama-3.1":
        substring = "<|start_header_id|>assistant<|end_header_id|>\n"
    else:
        substring = "</s>\n"
    pos = response.rfind(substring) + len(substring)
    response = response[pos:]
    response = response.replace("\n", "")
    response = response.replace(" ", "_")
    response = response.replace("\_", "_")

    return response

def init_pipeline(parameters, llm):
    if llm == "Llama-3.1":
        model_id = f"Meta-Llama-3.1-{parameters}B-Instruct"
    else:
        model_id = f"Mixtral-8x{parameters}B-Instruct-v0.1"

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    terminators = [tokenizer.eos_token_id]
    if llm == "Llama-3.1":
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=32,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        max_length=8192,
        truncation=True
    )
    if llm == "Llama-3.1":
        pad = model.config.eos_token_id[0]
    else:
        pad = model.config.eos_token_id
    pipe.tokenizer.pad_token_id = pad
    pipe = HuggingFacePipeline(pipeline=pipe)

    return pipe


def write_config(exp_name, method, mode, model, dataset, summarization, include_ranking, examples):
    config = {
        "method": method,
        "mode": mode,
        "model": model,
        "dataset": dataset,
        "summarization": summarization,
        "include_ranking": include_ranking,
        "examples": examples
    }

    write_json(config, FS_CONFIGS_PATH / f"{exp_name}.json")

@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option("--model", type=click.Choice(MODELS))
@click.option("--method", type=click.Choice(METHODS))
@click.option("--mode", type=click.Choice(MODES + ["None"]))
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS + ["None"]))
@click.option("--include-ranking", type=int)
@click.option("--include-examples", type=int)
@click.option("--parameters", type=int)
@click.option("--llm", type=click.Choice(["Llama-3.1", "Mixtral"]))
def main(dataset, model, method, mode, summarization, include_ranking, include_examples, parameters, llm):
    if mode == "None":
        mode = None
    if summarization == "None":
        summarization = None
    set_seeds(42)

    if llm == "Llama-3.1":
        prefix = prefix_llama
        suffix = suffix_llama
        example_prompt_str_template = example_prompt_str_template_llama
    else:
        prefix = prefix_mistral
        suffix = suffix_mistral
        example_prompt_str_template = example_prompt_str_template_mistral

    include_ranking = True if include_ranking == 1 else False
    examples = True if include_examples == 1 else False

    exp_name = f"{method}_{mode}_{model}_{dataset}_{summarization}_{'yes' if include_ranking else 'no'}_{'yes' if examples else 'no'}_{llm}_{parameters}B"

    write_config(exp_name, method, mode, model, dataset, summarization, include_ranking, examples)

    pipe = init_pipeline(parameters, llm)

    start = time.time()

    paths = format_paths(method, mode, model, dataset, summarization)

    lp_config_path = LP_CONFIGS_PATH / f"{model}_{dataset}.json"
    lp_config = read_json(lp_config_path)

    print(f"Loading KG {dataset}...")
    dataset = Dataset(dataset)
    print(f"Loading KGE model {model}...")
    model = load_model(lp_config, dataset)
    model.eval()
    evaluator = Evaluator(model=model)
    ranks = evaluator.evaluate(triples=dataset.training_triples)
    df_output = evaluator.get_df_output(triples=dataset.training_triples, ranks=ranks)
    example_selector = CustomExampleSelector(dataset, df_output)
    example_prompt_template = PromptTemplate(
        template=example_prompt_str_template, input_variables=["subject", "predicate", "object"]
    )

    few_shot_template = FewShotPromptTemplate(
        example_selector=example_selector if examples else None,
        examples=None if examples else [],
        example_prompt=example_prompt_template,
        input_variables=["subject", "predicate", "explanation", "ranking"],
        prefix=prefix,
        suffix=suffix,
        example_separator=""
    )

    chain = few_shot_template | pipe | (lambda x: parse(x, llm))

    rankings = read_json(paths["rankings"])
    rankings = {tuple(ranking["pred"]): ranking["ranking"][:100] for ranking in rankings}
    explained_preds = read_json(paths["exps"])

    explained_preds = [
        {
            "pred": explained_pred["pred"],
            "explanation": explained_pred["explanation"],
            "quotient_explanation": explained_pred.get("quotient_explanation", None),
            "label": explained_pred.get("label", None),
            "ranking": rankings[tuple(explained_pred["pred"])]
        }
        for explained_pred in explained_preds
    ]

    if method == "benchmark":
        explained_preds = [ep for ep in explained_preds if ep["label"] == 1]

    queries = [
        {
            "subject": explained_pred["pred"][0],
            "predicate": explained_pred["pred"][1],
            "explanation": "",
            "ranking": build_ranking(explained_pred["ranking"]) if include_ranking else "",
        }
        for explained_pred in explained_preds
    ]

    gts = [ep["pred"][2] for ep in explained_preds]
    print("Running simulations...")
    simulations = chain.batch(queries)

    e_sem = None

    if dataset.name in [DB50K, DB100K, YAGO4_20]:
        e_sem = pd.read_csv(
            DATA_PATH / dataset.name / "reasoned" / "entities.csv",
            converters={"classes": literal_eval},
        )
        e_sem["classes"] = e_sem["classes"].map(sorted)
        e_sem["classes"] = e_sem["classes"].map(lambda x: [c.split("/")[-1] for c in x])
        e_sem["classes"] = e_sem["classes"].map(", ".join)
        e_sem = e_sem.to_dict("records")
        e_sem = {e["entity"]: e["classes"] for e in e_sem}

    explained_queries = [
        {
            "subject": explained_pred["pred"][0],
            "predicate": explained_pred["pred"][1],
            "explanation": build_explanation(explained_pred, summarization, e_sem) if method != "GEnI" else explained_pred["explanation"],
            "ranking": build_ranking(explained_pred["ranking"]) if include_ranking else "",
        }
        for explained_pred in explained_preds
    ]

    print("Running post-explanation simulations...")
    post_exp_simulations = chain.batch(explained_queries)

    print("Writing results...")
    predictability_pre = [1 if o == gt else 0 for o, gt in zip(simulations, gts)]
    predictability_post = [1 if o == gt else 0 for o, gt in zip(post_exp_simulations, gts)]

    explanation_labels = [post - pre for post, pre in zip(predictability_post, predictability_pre)]

    for i in range(len(explained_preds)):
        explained_preds[i]["simulation"] = simulations[i]
        explained_preds[i]["post_exp_simulation"] = post_exp_simulations[i]
        explained_preds[i]["predictability_pre"] = predictability_pre[i]
        explained_preds[i]["predictability_post"] = predictability_post[i]
        explained_preds[i]["fs_label"] = explanation_labels[i]

    write_json(explained_preds, FS_DETAILS_PATH / f"{exp_name}.json")

    avg_pre = sum(predictability_pre) / len(predictability_pre)
    avg_post = sum(predictability_post) / len(predictability_post)

    end = time.time()

    metrics = {
        "avg_pre": avg_pre,
        "avg_post": avg_post,
        "delta": avg_post - avg_pre,
        "-1": str(len([l for l in explanation_labels if l == -1])),
        "0": str(len([l for l in explanation_labels if l == 0])),
        "1": str(len([l for l in explanation_labels if l == 1])),
        "total": str(len(explanation_labels)),
        "time": str(end - start)
    }

    write_json(metrics, FS_METRICS_PATH / f"{exp_name}.json")

if __name__ == "__main__":
    main()
