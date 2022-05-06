from argparse import ArgumentParser

from datasets import load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import pandas as pd
import numpy as np
import torch
import gc

from yaml import parse


def argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description="T5 training for ABSAPT")

    parser.add_argument("--test_data", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)

    # Training args
    parser.add_argument("--max_input_length", default=128, type=int, required=False)
    parser.add_argument("--max_target_length", default=64, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--evaluation_strategy", default="steps", required=False)
    parser.add_argument("--learning_rate", default=5e-05, type=float, required=False)
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--save_total_limit", default=2, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--fp16", default=False, type=bool)
    parser.add_argument("--chosen_metric", default="f1", required=False)

    return parser.parse_args()


def get_dataset(data_path):
    data = pd.read_csv(data_path, sep=";")

    text = "Review: " + data["review"] + "; Aspect: " + data["aspect"]
    output = data["polarity"].map({-1: "negativo", 0: "neutro", 1: "positivo"})

    data = pd.concat([text, output], axis=1)

    return data


def partial_preprocess_function(
    tokenizer, max_input_length, max_target_length, examples
):
    inputs = [(doc) for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        _labels = [(doc) for doc in examples["output"]]
        labels = tokenizer(_labels, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def partial_compute_metrics(tokenizer, metric, eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    label_dict = {"negativo": -1, "neutro": 0, "positivo": 1}

    decoded_labels = [label_dict[x] for x in decoded_labels]
    decoded_preds = [label_dict[x] for x in decoded_preds]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, average="macro"
    )

    return {k: v for k, v in result.items()}


def main():
    args = argument_parser()

    train_data = get_dataset(args.train_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    metric = load_metric(args.chosen_metric)
    compute_metrics = lambda eval_pred: partial_compute_metrics(
        tokenizer, metric, eval_pred
    )
    preprocess_function = lambda examples: partial_preprocess_function(
        tokenizer, args.max_input_length, args.max_target_length, examples
    )

    tokenized_datasets_train = train_data.map(preprocess_function, batched=True)

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=args.model_name_or_path,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_predit_batch_size=args.per_device_predict_batch_size,
        seed=args.seed,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=tokenized_datasets_train,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()


if __name__ == "__main__":
    main()
