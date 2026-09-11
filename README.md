# 🧠 Mood Mentor

### Emotion and Sentiment Analysis using BERT, DistilBERT and Multi-Label Classification

Mood Mentor is a Python-based Natural Language Processing (NLP) project designed to analyze text and identify **sentiment and emotional states**.

The project combines traditional sentiment analysis with transformer-based emotion classification to provide a more detailed understanding of a user's emotional state.

---

## 🌟 Project Overview

Traditional sentiment analysis generally classifies text as **Positive, Negative, or Neutral**.

However, human emotions are more complex. A single sentence can contain multiple emotions at the same time.

For example:

> "I am excited about my promotion but nervous about the extra responsibility."

This sentence contains both **Joy** and **Fear**.

Mood Mentor addresses this by combining:

* VADER sentiment analysis
* BERT emotion classification
* DistilBERT emotion classification
* Multi-label emotion classification
* Confidence scores
* ISEAR-based model validation
* Integrated Milestone 1 + Milestone 2 analysis

---

## 🎯 Objectives

The main objectives of Mood Mentor are:

1. Perform sentiment analysis on user-provided text.
2. Classify text into different emotional categories.
3. Compare BERT and DistilBERT emotion predictions.
4. Detect multiple emotions in a single input.
5. Generate confidence scores for emotion predictions.
6. Evaluate trained models using a held-out ISEAR benchmark.
7. Integrate sentiment and emotion classification into a single pipeline.
8. Generate structured emotion classification reports.

---

## ✨ Key Features

### 📝 1. Text Input

The system supports:

* Direct text input
* `.txt` file input
* `.csv` file input

Example:

```powershell
python task10_final_validation.py --text "I am extremely happy about my new job!"
```

---

### 💬 2. Sentiment Analysis

Mood Mentor uses **VADER** to classify text into:

* Positive
* Negative
* Neutral

It also generates:

* Compound score
* Positive score
* Negative score
* Neutral score

---

### 🤖 3. BERT Emotion Classification

A fine-tuned BERT model is used to classify text into six emotions:

* Joy
* Sadness
* Anger
* Fear
* Surprise
* Disgust

The model also provides a confidence score for the predicted emotion.

---

### ⚡ 4. DistilBERT Emotion Classification

A fine-tuned DistilBERT model is included for emotion classification.

DistilBERT provides a lighter transformer architecture while maintaining strong classification performance.

The system reports:

* Predicted emotion
* Confidence score

This also allows BERT and DistilBERT predictions to be compared.

---

### 🎭 5. Multi-Label Emotion Classification

Unlike single-label classification, the multi-label model can identify **multiple emotions in the same input**.

For example:

```text
Input:
I am excited about my promotion but nervous about the extra responsibility.

Detected emotions:
Fear, Joy, Surprise
```

The model generates independent scores for each emotion.

---

### 📊 6. Confidence Scores

The system reports confidence scores for:

* BERT
* DistilBERT
* Multi-label emotion predictions

This helps indicate how strongly a model supports a particular prediction.

---

### 📈 7. ISEAR Validation

The trained emotion models are evaluated using a held-out portion of the **ISEAR emotion dataset**.

The validation process generates:

* Accuracy
* Macro F1 score
* Benchmark sample count

The current Task 10 validation produced:

| Model      | Accuracy | Macro F1 |
| ---------- | -------: | -------: |
| BERT       |   0.7305 |   0.7255 |
| DistilBERT |   0.6972 |   0.6859 |

Benchmark size:

**1,169 samples**

---

## 🧩 Project Architecture

```text
                    ┌─────────────────────┐
                    │     User Input      │
                    │ Text / TXT / CSV    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Text Preprocessing │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ VADER Sentiment │        │ Transformer     │
        │    Analysis     │        │ Emotion Models  │
        └────────┬────────┘        └────────┬────────┘
                 │                          │
                 │              ┌───────────┼───────────┐
                 │              │           │           │
                 │              ▼           ▼           ▼
                 │            BERT      DistilBERT   Multi-Label
                 │              │           │           │
                 └──────────────┴───────────┴───────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Integrated Emotion  │
                    │ Classification      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ CSV / JSON Reports  │
                    └─────────────────────┘
```

---

## 🛠️ Technology Stack

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Main programming language          |
| NLP          | Text processing and analysis       |
| VADER        | Sentiment analysis                 |
| BERT         | Emotion classification             |
| DistilBERT   | Lightweight emotion classification |
| Transformers | Transformer model implementation   |
| PyTorch      | Deep learning framework            |
| Pandas       | Data processing                    |
| Scikit-learn | Evaluation metrics                 |
| ISEAR        | Emotion classification dataset     |
| Git & GitHub | Version control                    |

---

## 📁 Project Structure

```text
Mood-mentor/
│
├── .gitignore
├── LICENSE
├── README.md
│
├── milestone1.py
├── report_generator.py
├── preprocessing.py
├── sentiment_analysis.py
├── text_ingestion.py
├── pipeline.py
│
├── bert_model.py
├── distilbert_model.py
├── multi_label_classifier.py
├── emotion_labels.py
├── evaluate_model.py
├── isear_data.py
│
├── extract_surprise_from_goemotions.py
├── surprise_examples.csv
│
├── task6_benchmark_validation.py
├── task7_integration.py
├── task8_testing.py
├── task10_final_validation.py
├── verify_task4.py
│
├── test_suite.py
├── test_emotion_model.py
└── test_multi_label.py
```

### Generated / local-only files

Large trained models, checkpoints, downloaded datasets, Python cache files and generated reports are intentionally excluded from GitHub using `.gitignore`.

Examples:

```text
bert_emotion_model/
distilbert_emotion_model/
multilabel_emotion_model/

bert_emotion_checkpoints/
distilbert_emotion_checkpoints/

goemotions_data/
__pycache__/

ISEAR_spellchecked.csv
```

---

# 🚀 Installation

## 1. Clone the repository

```powershell
git clone https://github.com/priaanka01/Mood-mentor.git
```

## 2. Enter the project directory

```powershell
cd Mood-mentor
```

## 3. Create a virtual environment

```powershell
python -m venv .venv
```

## 4. Activate the virtual environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

## 5. Install dependencies

Install the required Python packages used by the project:

```powershell
pip install torch transformers pandas scikit-learn nltk
```

If VADER data has not been downloaded:

```powershell
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

# ▶️ Running the Project

## Milestone 1 – Sentiment Analysis

Run direct text analysis:

```powershell
python milestone1.py --text "I am very happy with my new job!"
```

The system generates sentiment information such as:

```text
Positive
Compound Score
Positive Score
Negative Score
Neutral Score
```

---

# 🤖 Milestone 2 – Emotion Classification

Mood Mentor uses three emotion classification approaches:

### BERT

```powershell
python bert_model.py --help
```

### DistilBERT

```powershell
python distilbert_model.py --help
```

### Multi-Label Classification

```powershell
python multi_label_classifier.py --help
```

Example:

```powershell
python multi_label_classifier.py --text "I am excited about my promotion but nervous about the extra responsibility."
```

---

# 🔗 Integrated Milestone 1 + Milestone 2

The integrated system combines:

```text
VADER Sentiment
       +
BERT Emotion
       +
DistilBERT Emotion
       +
Multi-Label Emotion
       ↓
Integrated Classification Report
```

Run:

```powershell
python task10_final_validation.py --text "I am extremely happy about my new job!"
```

Another example:

```powershell
python task10_final_validation.py --text "I am excited about my promotion but nervous about the extra responsibility."
```

---

# 📄 Input Options

Task 10 supports three input methods.

### 1. Direct text

```powershell
python task10_final_validation.py --text "I feel happy today."
```

### 2. TXT file

```powershell
python task10_final_validation.py --txt "input.txt"
```

### 3. CSV file

```powershell
python task10_final_validation.py --csv "input.csv"
```

If the CSV contains a specific text column:

```powershell
python task10_final_validation.py --csv "input.csv" --column "text"
```

---

# 📊 Final Classification Report

The integrated Task 10 pipeline generates:

```text
milestone2_final_report.csv
milestone2_final_report.json
```

The classification report contains information including:

* Original input text
* Processed text
* Sentiment label
* Compound sentiment score
* Positive score
* Negative score
* Neutral score
* BERT emotion
* BERT confidence
* DistilBERT emotion
* DistilBERT confidence
* Detected multi-label emotions
* Primary multi-label emotion
* Multi-label threshold
* Individual emotion scores

Generated reports are excluded from GitHub through `.gitignore`.

---

# 🧪 Model Validation

Task 10 verifies the following:

```text
✓ BERT model works
✓ DistilBERT model works
✓ Multi-label emotion classification works
✓ Confidence scores are generated
✓ Evaluation metrics are generated
✓ ISEAR validation is completed
✓ Milestone 1 integration works
✓ Final emotion classification report is generated
```

Run the complete validation with:

```powershell
python task10_final_validation.py --text "I am extremely happy about my new job!"
```

---

# 📈 Example Output

```text
Integrated Classification Results
----------------------------------------------------------------------

Input 1
Text: I am extremely happy about my new job!

Sentiment      : Positive
BERT           : Joy
DistilBERT     : Surprise
Multi-label    : Joy, Surprise
Primary Multi  : Joy
```

For a mixed-emotion input:

```text
Input:
I am excited about my promotion but nervous about the extra responsibility.

Sentiment      : Negative
BERT           : Joy
DistilBERT     : Joy
Multi-label    : Fear, Joy, Surprise
Primary Multi  : Fear
```

---

# 🔬 Milestones

## Milestone 1 – Sentiment Analysis

Completed:

* Text ingestion
* Text preprocessing
* VADER sentiment analysis
* Positive / Negative / Neutral classification
* Sentiment scoring
* CSV/JSON report generation

## Milestone 2 – Emotion Classification

Completed:

* BERT emotion classifier
* DistilBERT emotion classifier
* Multi-label emotion classifier
* Confidence scores
* ISEAR validation
* Model evaluation
* Milestone 1 + Milestone 2 integration
* Final emotion classification report
* Task 10 validation

---

# 🔐 Data and Model Files

Large model files and datasets are intentionally not stored in this Git repository.

The following are ignored:

```text
bert_emotion_model/
distilbert_emotion_model/
multilabel_emotion_model/

bert_emotion_checkpoints/
distilbert_emotion_checkpoints/

goemotions_data/
ISEAR_spellchecked.csv
```

This keeps the repository lightweight and focused on the source code.

---

# 🧪 Testing

The project contains testing and validation scripts including:

```text
test_suite.py
test_emotion_model.py
test_multi_label.py
task6_benchmark_validation.py
task7_integration.py
task8_testing.py
task10_final_validation.py
verify_task4.py
```

---

# 🔮 Future Scope

Possible future improvements include:

* Web-based Mood Mentor interface
* Real-time mood tracking
* Conversation history
* Personalized emotional insights
* Emotion trend visualization
* Improved multi-label emotion models
* Additional datasets for validation
* Explainable emotion predictions
* Voice-based emotion analysis
* Mobile application
* Deployment as an API or web application

---

# ⚠️ Disclaimer

Mood Mentor is an **academic and experimental NLP project**.

Its predictions are intended for educational and research purposes and should not be considered a medical or psychological diagnosis.

---

# 👩‍💻 Author

**Priaanka Avanigadda**

B.Tech – Computer Science and Engineering

GitHub:
https://github.com/priaanka01

---

# 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
