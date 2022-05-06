from argparse import ArgumentParser
import json
import os

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import pandas as pd


def argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description="T5 training for ABSAPT")

    parser.add_argument("--test_data", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_input_length", default=128, type=int, required=False)

    return parser.parse_args()


def get_dataset(data_path):
    data = pd.read_csv(data_path, sep=";")
    data = "Review: " + data["review"] + "; Aspect: " + data["aspect"]
    data = Dataset.from_pandas(data[["text"]])

    return data


def partial_preprocess_function(tokenizer, max_input_length, examples):
    inputs = [(doc) for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    return model_inputs


def predict(model, tokenizer, batch):
    input_ids = tokenizer.encode(
        batch, return_tensors="pt", add_special_tokens=True, padding=True
    ).to("cuda")

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

    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return predictions


def main():
    args = argument_parser()

    test_data = get_dataset(args.test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
    )

    preprocess_function = lambda examples: partial_preprocess_function(
        tokenizer, args.max_input_length, examples
    )
    tokenized_datasets_test = test_data.map(preprocess_function, batched=True)
    predictions = trainer.predict(tokenized_datasets_test)
    print(predictions)

    predictions_file = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_file, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
