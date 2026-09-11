"""
Task 2 - DistilBERT Model Integration
=========================================
Same structure and interface as bert_model.py (Task 1), swapped to
DistilBERT: Load DistilBERT -> Configure Tokenizer -> Fine-tune on the
same ISEAR data/split (isear_data.py, emotion_labels.py - unchanged
from Task 1, so both models are trained/evaluated on identical data
for a fair comparison) -> Save -> Predict -> (Task 2's own step)
Compare BERT vs DistilBERT results.

DistilBERT is ~40% smaller and roughly 60% faster than BERT-base at
inference/training time, with a small (typically a few points) drop
in accuracy - worth watching for given how slow BERT was to train on
your CPU-only machine. This may end up being the more practical
day-to-day model, with BERT-base as the accuracy reference point.

RUNTIME NOTE: same as bert_model.py - downloads distilbert-base-uncased
from huggingface.co on first use, so this must run in an environment
with open internet (not this dev sandbox).

Install once, in the environment where you actually run this:
    pip install torch transformers accelerate --break-system-packages
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from emotion_labels import EMOTION_LABELS, TRAINING_LABELS, merge_training_probs_softmax
from isear_data import EmotionRecord

BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 128


@dataclass
class EmotionPrediction:
    text: str
    primary_emotion: str
    primary_confidence: float
    all_scores: Dict[str, float]


class EmotionDistilBERTClassifier:
    """
    Thin wrapper around a HuggingFace DistilBertForSequenceClassification.
    Mirrors EmotionBERTClassifier's interface (bert_model.py) exactly,
    including support for either the 6-category output schema or the
    8-category native training schema (Shame/Guilt kept distinct - the
    Task 6 fix). .predict() always returns the required 6-category
    EmotionPrediction either way.
    """

    def __init__(self, model_name: str = BASE_MODEL_NAME, label_list: Optional[List[str]] = None):
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        self.label_list = label_list or EMOTION_LABELS
        label_to_index = {label: i for i, label in enumerate(self.label_list)}
        index_to_label = {i: label for label, i in label_to_index.items()}

        self.model_name = model_name
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
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
        import os
        import torch
        from torch.utils.data import Dataset
        from transformers import Trainer, TrainingArguments

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
            output_dir="./distilbert_emotion_checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=25,
            save_strategy="epoch",
            report_to=[],
        )
        trainer = Trainer(model=self.model, args=args, train_dataset=train_dataset)
        trainer.train()

    def save(self, output_dir: str = "./distilbert_emotion_model") -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[distilbert_model] Saved fine-tuned model + tokenizer to '{output_dir}'")

    @classmethod
    def load(cls, model_dir: str) -> "EmotionDistilBERTClassifier":
        """Restores whichever label schema (6-category or 8-category native) the saved model was trained with, from its own config."""
        instance = cls.__new__(cls)
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        instance.model_name = model_dir
        instance.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        instance.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        num_labels = instance.model.config.num_labels
        instance.label_list = [instance.model.config.id2label[i] for i in range(num_labels)]
        return instance

    def predict(self, text: str) -> EmotionPrediction:
        """
        Always returns the required 6-category EMOTION_LABELS output,
        merging Shame+Guilt back into Sadness automatically if this
        model was trained on the 8-category native schema.
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


def run_task2_demo(
    isear_csv_path: str,
    epochs: int = 3,
    max_train_rows: int = None,
    max_per_class: int = None,
    compare_with_bert_dir: str = None,
    native_training: bool = False,
) -> None:
    """
    Task 2's verification step: fine-tune DistilBERT on the same
    ISEAR split as Task 1, save, predict on the same sample sentences,
    and (if a saved BERT model directory is given) print both models'
    predictions side by side for Task 2's "Compare BERT and DistilBERT
    Results" step.

    `native_training=True` applies the same Task 6 fix as bert_model.py:
    trains on ISEAR's 8 native categories (Shame/Guilt kept distinct)
    instead of the pre-merged 6-category set.
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
        print(f"[distilbert_model] Capped to {max_per_class}/class: {len(train_records)} rows total {dict(Counter(r.label for r in train_records))}")
    elif max_train_rows is not None and len(train_records) > max_train_rows:
        import random
        random.Random(42).shuffle(train_records)
        train_records = train_records[:max_train_rows]
        print(f"[distilbert_model] Smoke-test mode: using {len(train_records)} of the full train set.")

    classifier = EmotionDistilBERTClassifier(BASE_MODEL_NAME, label_list=label_list)
    classifier.fine_tune(train_records, epochs=epochs)
    classifier.save("./distilbert_emotion_model")

    sample_texts = [
        "I am excited about the new opportunity but nervous about the outcome.",
        "My manager thanked me in front of the whole team out of nowhere.",
        "This workload is making me feel completely overwhelmed.",
    ]

    bert_classifier = None
    if compare_with_bert_dir:
        from bert_model import EmotionBERTClassifier
        bert_classifier = EmotionBERTClassifier.load(compare_with_bert_dir)

    for text in sample_texts:
        distil_pred = classifier.predict(text)
        print(f"\nText: {text}")
        print(f"  DistilBERT -> {distil_pred.primary_emotion} ({distil_pred.primary_confidence}) | {distil_pred.all_scores}")
        if bert_classifier:
            bert_pred = bert_classifier.predict(text)
            print(f"  BERT       -> {bert_pred.primary_emotion} ({bert_pred.primary_confidence}) | {bert_pred.all_scores}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task 2 - DistilBERT emotion classifier fine-tune/demo")
    parser.add_argument("isear_csv_path", help="Path to the ISEAR CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default 3)")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Blind cap on total training rows")
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Cap rows PER class (e.g. 300) - keeps classes balanced while cutting runtime.",
    )
    parser.add_argument(
        "--compare-with-bert-dir", type=str, default=None,
        help="Path to a saved BERT model (e.g. ./bert_emotion_model) to print side-by-side predictions.",
    )
    parser.add_argument(
        "--native-training", action="store_true",
        help="Task 6 fix: train on ISEAR's 8 native categories (Shame/Guilt kept distinct from "
             "Sadness) instead of the original pre-merged 6-category set.",
    )
    args = parser.parse_args()
    run_task2_demo(
        args.isear_csv_path,
        epochs=args.epochs,
        max_train_rows=args.max_train_rows,
        max_per_class=args.max_per_class,
        compare_with_bert_dir=args.compare_with_bert_dir,
        native_training=args.native_training,
    )