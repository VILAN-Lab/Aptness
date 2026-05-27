# LLM Evaluation Based on Metaphor Aptness

This project evaluates LLM performance based on **metaphor aptness**.

The script converts raw LLM answers into `Yes` / `No` labels, combines them with metaphor aptness labels, and calculates accuracy across different aptness levels.

## Directory Structure

```text
.
├── result.csv
├── VUA20_int_<MODEL_NAME>.csv
├── LLM_result/
│   ├── result_labeled_3.csv
│   ├── result_labeled_6.csv
│   ├── VUA20_aptness_<MODEL_NAME>.csv
│   └── VUA20_aptness_<MODEL_NAME>_labeled.csv
└── script.py
```

## Requirements

```bash
pip install pandas numpy
```

## Usage

### 1. Generate Metaphor Aptness Labels

Generate 3-class labels:

```python
get_new_label_by_aptness_rate(label_num=3)
```

Generate 6-class labels:

```python
get_new_label_by_aptness_rate(label_num=6)
```

Output files:

```text
LLM_result/result_labeled_3.csv
LLM_result/result_labeled_6.csv
```

### 2. Run LLM Evaluation

```python
Run_aptness_accuracy()
```

This function evaluates all models in `MODEL_NAME_LIST` under both 3-class and 6-class metaphor aptness settings.

## Input Files

| File | Description |
|---|---|
| `result.csv` | Original file containing metaphor aptness scores |
| `VUA20_int_<MODEL_NAME>.csv` | Raw LLM prediction file for each model |

## Output Files

| File | Description |
|---|---|
| `LLM_result/result_labeled_3.csv` | Metaphor aptness labels using 3 classes |
| `LLM_result/result_labeled_6.csv` | Metaphor aptness labels using 6 classes |
| `LLM_result/VUA20_aptness_<MODEL_NAME>.csv` | Normalized LLM answers |
| `LLM_result/VUA20_aptness_<MODEL_NAME>_labeled.csv` | LLM results combined with metaphor aptness labels |

## Label Settings

### 3-Class Labels

| Aptness Score Range | Class |
|---|---|
| `1 <= score < 5` | 0 |
| `5 <= score < 7` | 1 |
| `7 <= score <= 10` | 2 |

### 6-Class Labels

| Aptness Score Range | Class |
|---|---|
| `0 <= score <= 4.0` | 0 |
| `4.333 < score <= 5.0` | 1 |
| `5.0 < score < 6.0` | 2 |
| `6.0 <= score < 7.0` | 3 |
| `7 <= score <= 7.333` | 4 |
| `7.33 < score` | 5 |

Invalid or unexpected scores are assigned to `-1`.

## Accuracy Output

The evaluation result is printed as a table:

| Column | Description |
|---|---|
| `Level` | Metaphor aptness level |
| `Accuracy` | Accuracy for the level |
| `Sample Count` | Number of samples in the level |
