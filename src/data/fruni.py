from .. import EXPLANATIONS_PATH, SAMPLED_SELECTED_PREDICTIONS_PATH

from ..utils import write_json

def filter_rows(input_file):
    explanations = []
    with open(input_file, 'r') as infile:
        for line in infile:
            values = line.strip().split(',')
            if len(values) > 3:
                explanations.append(values)

    return explanations

input_file = 'data/FRUNI/test_explanations.txt'
output_file = EXPLANATIONS_PATH / f"benchmark_{None}_ComplEx_FRUNI_no_first.json"

explained_preds = filter_rows(input_file)

preds = SAMPLED_SELECTED_PREDICTIONS_PATH / "ComplEx_FRUNI_first.csv"
with open(preds, "r") as preds:
    preds = [x.strip().split("\t") for x in preds.readlines()]
preds = [(s, p, o) for s, p, o in preds]

explained_preds = [
    {
        "pred": (ep[0], ep[1], ep[2]),
        "explanation": [(ep[i], ep[i + 1], ep[i + 2]) for i in range(3, len(ep), 3)],
        "label": "1",
    }
    for ep in explained_preds if (ep[0], ep[1], ep[2]) in preds
]

print(len(explained_preds))

write_json(explained_preds, output_file)
