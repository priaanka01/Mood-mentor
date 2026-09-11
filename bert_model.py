"""
Task 1 - BERT Model Integration
===================================
Wraps a pretrained BERT checkpoint for emotion classification:
  Load Pre-trained BERT Model -> Configure Tokenizer ->
  Fine-tune on ISEAR (isear_data.py) -> Save -> Predict on sample text.

RUNTIME NOTE: loading `bert-base-uncased` downloads weights/tokenizer
files from huggingface.co on first use. This dev sandbox's network is
locked to a small allowlist that does not include huggingface.co, so
this module can be reviewed/edited here but must actually be RUN in
an environment with open internet (your local machine, Springboard,
or Colab). Everything in isear_data.py (no network needed) has
already been tested in-sandbox.

Install once, in the environment where you actually run this:
    pip install torch transformers --break-system-packages
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from emotion_labels import EMOTION_LABELS, TRAINING_LABELS, merge_training_probs_softmax
from isear_data import EmotionRecord

BASE_MODEL_NAME = "bert-base-uncased"
MAX_SEQUENCE_LENGTH = 128


@dataclass
class EmotionPrediction:
    text: str
    primary_emotion: str
    primary_confidence: float
    all_scores: Dict[str, float]  # every one of the 6 emotions -> probability


class EmotionBERTClassifier:
    """
    Thin wrapper around a HuggingFace BertForSequenceClassification.
    Can be configured for either the project's 6-category output
    schema (EMOTION_LABELS, the original Task 1 default) or the
    8-category native training schema (TRAINING_LABELS, Shame/Guilt
    kept distinct - the Task 6 fix). Whichever was used, .predict()
    always returns the required 6-category EmotionPrediction, merging
    Shame+Guilt back into Sadness automatically when needed.
    """

    def __init__(self, model_name: str = BASE_MODEL_NAME, label_list: Optional[List[str]] = None):
        # Imported lazily so this module can still be parsed/reviewed
        # in environments (like this sandbox) without torch/transformers
        # installed - the import only actually happens when a
        # classifier is instantiated to do real work.
        from transformers import BertTokenizerFast, BertForSequenceClassification

        self.label_list = label_list or EMOTION_LABELS
        label_to_index = {label: i for i, label in enumerate(self.label_list)}
        index_to_label = {i: label for label, i in label_to_index.items()}

        self.model_name = model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=index_to_label,
            label2id=label_to_index,
        )

    def fine_tune(
        self,
        train_records: List[EmotionRecord],
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> None:
        """
        Fine-tune on ISEAR training records (single-label per row -
        standard cross-entropy classification head). Uses HuggingFace
        Trainer so the loop itself (batching, optimizer, scheduler)
        isn't hand-rolled and re-debugged here.
        """
        import torch
        from torch.utils.data import Dataset
        from transformers import Trainer, TrainingArguments

        # On CPU-only machines, PyTorch doesn't always use every
        # available core by default. Explicitly using all of them can
        # meaningfully cut per-step time (you have 8 cores available -
        # worth using them all rather than leaving most idle).
        import os
        torch.set_num_threads(os.cpu_count())

        tokenizer = self.tokenizer
        label_to_index = {label: i for i, label in enumerate(self.label_list)}

        class _ISEARTorchDataset(Dataset):
            def __init__(self, records: List[EmotionRecord]):
                self.encodings = tokenizer(
                    [r.text for r in records],
                    truncation=True,
                    padding=True,
                    max_length=MAX_SEQUENCE_LENGTH,
                )
                self.labels = [label_to_index[r.label] for r in records]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        train_dataset = _ISEARTorchDataset(train_records)

        args = TrainingArguments(
            output_dir="./bert_emotion_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=25,
            save_strategy="epoch",
            report_to=[],  # no wandb/tensorboard side-effects
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=train_dataset)
        trainer.train()

    def save(self, output_dir: str = "./bert_emotion_model") -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[bert_model] Saved fine-tuned model + tokenizer to '{output_dir}'")

    @classmethod
    def load(cls, model_dir: str) -> "EmotionBERTClassifier":
        """
        Load a previously fine-tuned model from disk (skips the
        pretrained-base download). Automatically restores whichever
        label schema (6-category or 8-category native) the saved model
        was trained with, from its own config - HuggingFace persists
        id2label/label2id in config.json on save, so this doesn't need
        to be told which schema to expect.
        """
        instance = cls.__new__(cls)
        from transformers import BertTokenizerFast, BertForSequenceClassification

        instance.model_name = model_dir
        instance.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        instance.model = BertForSequenceClassification.from_pretrained(model_dir)
        num_labels = instance.model.config.num_labels
        instance.label_list = [instance.model.config.id2label[i] for i in range(num_labels)]
        return instance

    def predict(self, text: str) -> EmotionPrediction:
        """
        Run inference on a single piece of text. Confidence scores are
        computed dynamically via softmax over the model's real logits -
        nothing here is a hardcoded or placeholder value (Task 4
        requirement carried forward from this module too).

        Always returns the required 6-category EMOTION_LABELS output,
        regardless of whether this model was trained on the original
        6-category schema or the 8-category native schema (Shame/Guilt
        distinct) - the merge back into Sadness (if needed) happens
        here, transparently, via emotion_labels.merge_training_probs_softmax.
        """
        import torch

        self.model.eval()
        inputs = self.tokenizer(
            text, truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()

        raw_scores = {self.label_list[i]: round(probs[i], 4) for i in range(len(self.label_list))}
        all_scores = merge_training_probs_softmax(raw_scores) if set(self.label_list) != set(EMOTION_LABELS) else raw_scores

        top_label = max(all_scores.items(), key=lambda kv: kv[1])[0]
        return EmotionPrediction(
            text=text,
            primary_emotion=top_label,
            primary_confidence=all_scores[top_label],
            all_scores=all_scores,
        )


def run_task1_demo(
    isear_csv_path: str,
    epochs: int = 3,
    max_train_rows: int = None,
    max_per_class: int = None,
    native_training: bool = False,
) -> None:
    """
    Task 1's final verification step: fine-tune on ISEAR, save the
    model, then test it with sample text and confirm real predictions
    come out. Run this in an internet-connected environment.

    `epochs` / `max_train_rows` let you run a fast blind smoke test.
    `max_per_class` is the better option for a real (but faster) run
    on CPU-only machines: it caps rows per class rather than blindly
    truncating, so the trimmed set stays balanced (see
    isear_data.cap_rows_per_class).

    `native_training=True` is the Task 6 fix: trains on ISEAR's 8
    native categories (Shame/Guilt kept distinct from Sadness) instead
    of the original pre-merged 6-category training set. Predictions
    still come back in the standard 6-category form either way -
    EmotionBERTClassifier.predict() handles the merge automatically.
    """
    if native_training:
        from isear_data import prepare_isear_training_dataset, cap_rows_per_class
        train_records, _benchmark_records = prepare_isear_training_dataset(isear_csv_path)
        label_list = TRAINING_LABELS
    else:
        from isear_data import prepare_isear_dataset, cap_rows_per_class
        train_records, _benchmark_records = prepare_isear_dataset(isear_csv_path)
        label_list = EMOTION_LABELS

    if max_per_class is not None:
        train_records = cap_rows_per_class(train_records, max_per_class)
        from collections import Counter
        print(f"[bert_model] Capped to {max_per_class}/class: {len(train_records)} rows total {dict(Counter(r.label for r in train_records))}")
    elif max_train_rows is not None and len(train_records) > max_train_rows:
        import random
        random.Random(42).shuffle(train_records)
        train_records = train_records[:max_train_rows]
        print(f"[bert_model] Smoke-test mode: using {len(train_records)} of the full train set.")

    classifier = EmotionBERTClassifier(BASE_MODEL_NAME, label_list=label_list)
    classifier.fine_tune(train_records, epochs=epochs)
    classifier.save("./bert_emotion_model")

    sample_texts = [
        "I am excited about the new opportunity but nervous about the outcome.",
        "My manager thanked me in front of the whole team out of nowhere.",
        "This workload is making me feel completely overwhelmed.",
    ]
    for text in sample_texts:
        pred = classifier.predict(text)
        print(f"\nText: {text}")
        print(f"  Primary emotion : {pred.primary_emotion} ({pred.primary_confidence})")
        print(f"  All scores      : {pred.all_scores}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 1 - BERT emotion classifier fine-tune/demo")
    parser.add_argument("isear_csv_path", help="Path to the ISEAR CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default 3)")
    parser.add_argument(
        "--max-train-rows", type=int, default=None,
        help="Blind cap on total training rows for a fast smoke test (e.g. 500). Omit for full run.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Cap rows PER class (e.g. 300) - keeps classes balanced while cutting runtime. "
             "Recommended over --max-train-rows for a real (not just smoke-test) CPU run.",
    )
    parser.add_argument(
        "--native-training", action="store_true",
        help="Task 6 fix: train on ISEAR's 8 native categories (Shame/Guilt kept distinct from "
             "Sadness) instead of the original pre-merged 6-category set. Predictions are still "
             "reported in the standard 6-category form.",
    )
    args = parser.parse_args()
    run_task1_demo(
        args.isear_csv_path,
        epochs=args.epochs,
        max_train_rows=args.max_train_rows,
        max_per_class=args.max_per_class,
        native_training=args.native_training,
    )