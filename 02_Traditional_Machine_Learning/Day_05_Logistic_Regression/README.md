# Day 5: Classification

## Student Practice Notebook

**Programme:** Educentriks – Applied Data Science, AI/ML  
**Dataset:** T-ECD Retail Classroom  
**Notebook:** `Day_05_Classification_Student_Practice.ipynb`  
**Environment:** Python 3.13.x, Jupyter Notebook, `.venv`

## Overview

This session introduces classification as a supervised machine-learning task. Students use the approved T-ECD Retail Classroom dataset to classify the recorded `action_type` using selected attributes from the same event record.

The exercise is designed to introduce a practical classification workflow, model evaluation and the implications of class imbalance. It does not attempt to predict future customer behaviour or establish customer intent.

## Learning Outcomes

By the end of this practical, students should be able to:

- Explain classification as a supervised learning problem.
- Identify `action_type` as the target variable.
- Prepare categorical and time-derived features.
- Create stratified training and test partitions.
- Train a majority-class baseline and a Logistic Regression classifier.
- Evaluate a multiclass classifier using accuracy, precision, recall, F1-score and a confusion matrix.
- Explain why accuracy alone can be misleading when classes are imbalanced.

## Dataset

The notebook uses the approved **T-ECD Retail Classroom** subset.

Expected local structure:

```text
data/
└── T-ECD_Retail_Classroom/
    ├── metadata/
    └── [14 daily event-partition Parquet files]
```

The notebook searches recursively for Parquet files and reads these documented columns:

| Column | Use in this notebook |
|---|---|
| `timestamp` | Used to derive hour and day of week |
| `user_id` | Excluded as a raw identifier |
| `item_id` | Excluded as a raw identifier |
| `subdomain` | Categorical predictor |
| `action_type` | Classification target |
| `os` | Categorical predictor |

The notebook creates a reproducible sample of up to 100,000 records to keep the practical manageable on typical student laptops.

**Dataset licence:** T-ECD Retail is licensed under CC BY-NC-SA 4.0. Preserve the required attribution and licence information when redistributing permitted material. The course code licence does not replace or override the dataset licence.

## Software Requirements

- Python 3.13.x (64-bit)
- Jupyter Notebook
- Project virtual environment (`.venv`)

Install the required packages from the project environment:

```powershell
python -m pip install pandas pyarrow numpy matplotlib scikit-learn
```

## How to Run

1. Clone or download the student repository.
2. Ensure the T-ECD Retail Classroom dataset has been extracted into `data/T-ECD_Retail_Classroom/`.
3. Open the project in VS Code or Jupyter Notebook.
4. Activate the project's `.venv` environment.
5. Open `Day_05_Classification_Student_Practice.ipynb`.
6. Run the notebook from top to bottom.

If your dataset is stored elsewhere, update the `DATA_DIR` variable near the beginning of the notebook.

## Practical Workflow

The notebook guides students through:

1. Locating and loading the Parquet partitions.
2. Validating the schema and inspecting missing values.
3. Examining the distribution of `action_type`.
4. Visualising class imbalance.
5. Creating a reproducible modelling sample.
6. Deriving event hour and day-of-week features.
7. Creating a stratified training/test split.
8. Building preprocessing pipelines for categorical and numerical features.
9. Establishing a majority-class baseline with `DummyClassifier`.
10. Training a `LogisticRegression` classifier with balanced class weights.
11. Evaluating accuracy, precision, recall, macro F1 and weighted F1.
12. Comparing the classifier against the baseline.
13. Interpreting the confusion matrix.
14. Completing the hands-on challenges and knowledge check.

## Important Modelling Notes

- `action_type` is the target variable.
- `subdomain`, `os`, event hour and event day of week are the selected predictors.
- Raw `user_id` and `item_id` fields are excluded from this introductory model.
- The split is stratified to approximately preserve class proportions.
- Preprocessing is fitted inside a scikit-learn pipeline using training data.
- The test partition is reserved for final evaluation.
- `class_weight="balanced"` is an experiment to address class imbalance; it does not guarantee better performance.
- Do not interpret the exercise as a future-event prediction system. The target is the recorded action on the same event record.

## Student Submission Checklist

- [ ] Notebook runs from beginning to end.
- [ ] All expected T-ECD Parquet partitions are found and loaded.
- [ ] The target distribution is inspected.
- [ ] The reproducible modelling sample is created.
- [ ] Raw user and item identifiers are excluded from the baseline features.
- [ ] Training and test data are split using stratification.
- [ ] Majority-class baseline is evaluated.
- [ ] Logistic Regression classifier is trained and evaluated.
- [ ] Accuracy, precision, recall, F1 and confusion matrix are reviewed.
- [ ] Hands-on challenges are attempted.
- [ ] Business interpretation acknowledges class imbalance and model limitations.

## Repository Location

Save this README and the student notebook in:

```text
02_Traditional_Machine_Learning/
└── Day_05_Classification/
    ├── README.md
    └── Day_05_Classification_Student_Practice.ipynb
```

## Next Session

Continue with the next classification algorithm in the approved course sequence.
