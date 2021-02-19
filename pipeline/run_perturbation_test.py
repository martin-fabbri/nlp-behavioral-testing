import json

import nlpaug.augmenter.char as nac
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.tokenization_utils import PreTrainedTokenizer

from config import Config

transformer_model = "distilbert-base-uncased-finetuned-sst-2-english"

# load model
tokenizer = AutoTokenizer.from_pretrained(transformer_model)
inference_model = AutoModelForSequenceClassification.from_pretrained(transformer_model)

model = pipeline("sentiment-analysis", model=inference_model, tokenizer=tokenizer)

# define text perturbation
keyboard_aug = nac.KeyboardAug(aug_word_max=1)


def typo(aug, input):
    output = aug.augment(input)
    return output


def eval_perturb(input_a, input_b):
    output_a, output_b = model([input_a, input_b])
    sq_error = (output_a["score"] - output_b["score"]) ** 2
    acc = output_a["label"] == output_b["label"]
    # print(input_a, input_b)
    # print(output_a["label"], output_b["label"])
    # print("---")
    return sq_error, acc, output_b["score"]


# read in our test dataset
f = open(Config.TEST_SET)
test_dataset = f.read().split("\t")[:-1]

# Loop over all test examples and evaluate
mse, total_acc = 0, 0
n = len(test_dataset)
interesting_cases = []
for sentence in test_dataset:
    sentence_mod = typo(keyboard_aug, sentence)
    sq_error, acc, perturb_score = eval_perturb(sentence, sentence_mod)
    mse += (1 / n) * sq_error
    total_acc += (1 / n) * acc
    if acc == False:
        interesting_cases.append((sentence, sentence_mod, perturb_score))

interesting_cases.sort(key=lambda tup: tup[2], reverse=True)

# Write out our favorite interesting cases
to_report = interesting_cases[:5]
df = pd.DataFrame(to_report, columns=["Original", "Perturbed", "Model confidence"])
with open(Config.TOP_PERTURBATIONS, "w") as outfile:
    outfile.write(df.to_markdown(index=False))

# Write results to file
with open(Config.TEST_SCORES, "w") as outfile:
    json.dump({"accuracy": total_acc, "mse": mse}, outfile)
