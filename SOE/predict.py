from argparse import ArgumentParser
import json
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from tqdm import tqdm
import pandas as pd


def argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description="T5 training for ABSAPT")

    parser.add_argument("--test_data", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", default=32, type=int, required=False)

    return parser.parse_args()


def get_dataset(data_path):
    data = pd.read_csv(data_path, sep=";")
    data["text"] = "Review: " + data["review"] + "; Aspect: " + data["aspect"]
    data = data["text"].tolist()

    return data


def partial_predict(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(
        "cuda"
    )
    model.eval()

    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=32,
        min_length=1,
        early_stopping=True,
        repetition_penalty=2.5,
        do_sample=False,
        top_k=50,
        num_return_sequences=1,
    ).squeeze()

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    args = argument_parser()

    test_data = get_dataset(args.test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    predictions = list()
    predict = lambda batch: partial_predict(model, tokenizer, batch)
    for text in tqdm(test_data, desc="Generating.."):
        prediction = predict(text)
        predictions.append(prediction)

    predictions_file = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_file, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
