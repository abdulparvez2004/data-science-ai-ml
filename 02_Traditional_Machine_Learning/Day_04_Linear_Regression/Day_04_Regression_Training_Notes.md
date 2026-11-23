# Day 4: Regression – Building Predictive Models

## 1. Day Overview & Objectives

### Session Overview

Regression is one of the fundamental supervised machine-learning techniques used to predict continuous numerical outcomes. Organisations use regression models to estimate delivery times, sales revenue, product demand, operating costs and customer lifetime value.

On Day 3, we established the basic machine-learning workflow, explored training and testing partitions, and built a baseline using scikit-learn. Today, we move from a baseline model to actual regression models that learn relationships between input features and a numerical target.

**Approved dataset:** TabReD Delivery ETA  
**Learning environment:** Python 3.13.x, Jupyter Notebook, VS Code  
**Primary libraries:** NumPy, pandas, Matplotlib, scikit-learn  
**Session duration:** 90 minutes

### Core Learning Objectives

By the end of this session, students should be able to:

1. Explain regression and identify business problems that require regression models.
2. Understand the relationship between independent variables (features) and a dependent variable (target).
3. Build and train Linear Regression and Ridge Regression models using scikit-learn.
4. Evaluate regression models using MAE, MSE, RMSE and R².
5. Interpret model performance and communicate prediction errors in business terms.

### Prerequisites

Students should understand:

- Python functions, NumPy and pandas.
- Basic descriptive statistics.
- Features and target variables.
- Training, validation and test datasets.
- The scikit-learn Pipeline introduced on Day 3.

---

## 2. Conceptual Deep Dive

### 2.1 What Is Regression?

Regression is a supervised machine-learning technique used to predict a continuous numerical value based on one or more input variables.

Examples:

| Business Problem | Regression Target |
|---|---|
| Predict house prices | Property price |
| Forecast monthly sales | Sales revenue |
| Estimate delivery performance | Delivery-related numerical outcome |
| Predict employee compensation | Salary |
| Estimate electricity consumption | Energy usage |
| Predict customer lifetime value | Customer value |

The important characteristic is that the predicted outcome is numerical and continuous.

**Business perspective:** A retailer may want to estimate next month's sales for each store. A logistics company may want to estimate a delivery-related outcome to improve planning and customer communication.

### 2.2 Regression vs Classification

Both are supervised machine-learning techniques, but they solve different prediction problems.

| Regression | Classification |
|---|---|
| Predicts numerical values | Predicts categories or classes |
| Output may be 125.50 or 42.7 | Output may be Yes/No |
| Predicts sales revenue | Predicts whether a customer will churn |
| Predicts a numerical delivery outcome | Predicts whether delivery will be delayed |

A useful question to ask before selecting an algorithm is:

**Are we predicting a numerical quantity or assigning an observation to a category?**

### 2.3 Understanding Features and Target

Consider a simplified sales prediction problem.

| Advertising Spend | Store Size | Previous Sales | Actual Sales |
|---:|---:|---:|---:|
| 10,000 | 1,200 | 45,000 | 51,000 |
| 15,000 | 1,500 | 52,000 | 63,000 |
| 8,000 | 900 | 30,000 | 34,000 |

Here:

- Independent variables (X): Advertising Spend, Store Size and Previous Sales.
- Dependent variable (y): Actual Sales.

The model learns relationships between X and y from historical observations.

### 2.4 Understanding Linear Regression

Linear Regression models the relationship between input variables and a continuous target.

For a single input variable, the equation is:

\[
y = \beta_0 + \beta_1x + \epsilon
\]

Where:

- \(y\): Predicted target value.
- \(\beta_0\): Intercept.
- \(\beta_1\): Regression coefficient.
- \(x\): Input feature.
- \(\epsilon\): Error term.

For multiple features:

\[
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n
\]

The model estimates coefficients that minimise the sum of squared residuals during ordinary least squares fitting.

### 2.5 Understanding Residuals

A residual is the difference between the actual value and the predicted value.

\[
Residual = Actual - Predicted
\]

Example:

| Actual Sales | Predicted Sales | Residual |
|---:|---:|---:|
| 50,000 | 48,000 | 2,000 |
| 60,000 | 63,000 | -3,000 |
| 45,000 | 44,500 | 500 |

Residual analysis helps us understand where a model makes errors and whether those errors follow a pattern.

A model that consistently underestimates high-value observations may not be reliable for business planning.

### 2.6 Multiple Linear Regression

In real business problems, outcomes are usually influenced by several variables.

For example, sales may depend on:

- Advertising expenditure.
- Product price.
- Store location.
- Seasonality.
- Historical sales.
- Promotional activity.

Multiple Linear Regression incorporates several input variables to estimate the target.

The model estimates the contribution of each feature while accounting for the other included features, subject to the assumptions and limitations of the model.

### 2.7 Regularisation: Why Ridge Regression?

Ordinary Linear Regression can become unstable when features are highly correlated or when a model has many input variables.

Ridge Regression introduces an L2 penalty on coefficient magnitudes.

\[
\text{Ridge Objective} =
\text{RSS} + \alpha \sum_{j=1}^{p}\beta_j^2
\]

Where:

- RSS: Residual Sum of Squares.
- \(\alpha\): Regularisation strength.
- \(\beta_j\): Model coefficients.

As the regularisation strength increases, coefficients are pushed towards zero. Ridge Regression does not normally make coefficients exactly zero.

**Business relevance:** When a model uses many related operational variables, regularisation can help control coefficient instability and improve generalisation.

The value of \(\alpha\) should be selected using training and validation data, not by repeatedly tuning against the test set.

---

## 3. Regression Evaluation Metrics

A regression model should not be evaluated only by looking at a few predictions. We need numerical metrics to understand its errors.

### 3.1 Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and predicted values.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

**Interpretation:** If MAE is 5, the average absolute prediction error is 5 target units.

MAE is relatively easy to explain to business stakeholders.

### 3.2 Mean Squared Error (MSE)

MSE calculates the average squared prediction error.

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Large errors receive greater weight because the residuals are squared.

### 3.3 Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE is expressed in the same units as the target and penalises large errors more strongly than MAE.

The TabReD dataset metadata specifies RMSE as its scoring metric.

### 3.4 R² Score

R² measures how much of the variation in the target is explained by the model relative to a mean-prediction baseline.

\[
R^2 = 1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
\]

General interpretation:

| R² Value | General Interpretation |
|---:|---|
| 1.0 | Perfect predictions on the evaluated data |
| 0.8 | Strong explanatory performance in many contexts |
| 0.5 | Some variation explained |
| 0.0 | No improvement over predicting the evaluation-set mean |
| Below 0 | Worse than the evaluation-set mean baseline |

R² is not a percentage accuracy score. A negative R² is possible, particularly on unseen data.

**Important:** No single metric tells the complete story. MAE, RMSE and R² should be interpreted together and in the context of the business problem.

---

## 4. Practical Implementation & Code

### 4.1 Dataset and Modelling Approach

We will continue using the approved TabReD Delivery ETA dataset.

The inspected dataset structure is:

| Array | Shape | Role |
|---|---:|---|
| `x_num.npy` | (416451, 221) | Numerical features |
| `x_cat.npy` | (416451, 1) | Integer-coded categorical feature |
| `x_bin.npy` | (416451, 1) | Binary feature |
| `x_meta.npy` | (416451, 2) | Metadata |
| `y.npy` | (416451,) | Regression target |

For this introductory regression exercise:

- Use the 221 numerical features and one binary feature.
- Exclude `x_cat` until its category mapping is documented.
- Exclude `x_meta` until its meaning is documented.
- Use the official `default` training, validation and test indices.
- Do not create a new random split.
- Do not assign the 65,935 rows not referenced by the supplied default split to another partition.

The target's business unit and transformation have not been established by the metadata inspected so far. Therefore, interpret all errors in target units without calling them minutes or hours.

### 4.2 Import Libraries

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

### 4.3 Load the Dataset

```python
# Set the dataset directory.
DATA_DIR = Path("data/TabReD_Delivery_ETA/delivery-eta")

# Load numerical, binary and target arrays.
x_num = np.load(DATA_DIR / "x_num.npy", mmap_mode="r")
x_bin = np.load(DATA_DIR / "x_bin.npy", mmap_mode="r")
y = np.load(DATA_DIR / "y.npy", mmap_mode="r")

# Combine the numerical and binary features.
X = np.concatenate(
    [
        np.asarray(x_num, dtype=np.float32),
        np.asarray(x_bin, dtype=np.float32)
    ],
    axis=1
)

y = np.asarray(y, dtype=np.float32)

print("Feature matrix:", X.shape)
print("Target vector:", y.shape)
```

Expected shapes:

```text
Feature matrix: (416451, 222)
Target vector: (416451,)
```

### 4.4 Load the Official Dataset Splits

```python
# Load the supplied default split indices.
SPLIT_DIR = DATA_DIR / "splits" / "default"

train_idx = np.load(SPLIT_DIR / "train.npy")
val_idx = np.load(SPLIT_DIR / "val.npy")
test_idx = np.load(SPLIT_DIR / "test.npy")

# Create training, validation and test partitions.
X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)
```

### 4.5 Build a Linear Regression Pipeline

A pipeline keeps preprocessing and modelling steps together.

```python
# Create a Linear Regression pipeline.
linear_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]
)

# Train the model using only the training partition.
linear_pipeline.fit(X_train, y_train)

print("Linear Regression model trained successfully.")
```

### 4.6 Generate Predictions

```python
# Generate predictions for training and validation data.
train_predictions = linear_pipeline.predict(X_train)
val_predictions = linear_pipeline.predict(X_val)

# Display a sample of actual and predicted values.
comparison = pd.DataFrame({
    "Actual": y_val[:10],
    "Predicted": val_predictions[:10]
})

comparison["Residual"] = (
    comparison["Actual"] - comparison["Predicted"]
)

display(comparison)
```

### 4.7 Evaluate the Regression Model

```python
# Calculate regression evaluation metrics.
mae = mean_absolute_error(y_val, val_predictions)

mse = mean_squared_error(y_val, val_predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_val, val_predictions)

print("VALIDATION PERFORMANCE")
print("----------------------")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
```

Students should run the notebook and record the actual results. Do not use assumed or illustrative scores as measured dataset results.

### 4.8 Compare Training and Validation Performance

```python
# Calculate metrics on the training partition.
train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = np.sqrt(
    mean_squared_error(y_train, train_predictions)
)
train_r2 = r2_score(y_train, train_predictions)

# Create a performance comparison.
performance = pd.DataFrame({
    "Partition": ["Training", "Validation"],
    "MAE": [
        train_mae,
        mean_absolute_error(y_val, val_predictions)
    ],
    "RMSE": [
        train_rmse,
        np.sqrt(mean_squared_error(y_val, val_predictions))
    ],
    "R2": [
        train_r2,
        r2_score(y_val, val_predictions)
    ]
})

display(performance)
```

Discuss:

- Is the validation error substantially higher than training error?
- Does the model generalise reasonably?
- Are the errors acceptable for the intended business use?
- What additional model or feature investigation might be needed?

### 4.9 Train a Ridge Regression Model

```python
# Create a Ridge Regression pipeline.
ridge_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]
)

# Fit the model on training data only.
ridge_pipeline.fit(X_train, y_train)

# Generate validation predictions.
ridge_predictions = ridge_pipeline.predict(X_val)

# Evaluate Ridge Regression.
ridge_mae = mean_absolute_error(y_val, ridge_predictions)

ridge_rmse = np.sqrt(
    mean_squared_error(y_val, ridge_predictions)
)

ridge_r2 = r2_score(y_val, ridge_predictions)

print("RIDGE REGRESSION VALIDATION PERFORMANCE")
print("---------------------------------------")
print(f"MAE:  {ridge_mae:.4f}")
print(f"RMSE: {ridge_rmse:.4f}")
print(f"R²:   {ridge_r2:.4f}")
```

### 4.10 Compare the Models

```python
model_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MAE": [mae, ridge_mae],
    "RMSE": [rmse, ridge_rmse],
    "R2": [r2, ridge_r2]
})

display(model_comparison)
```

Discuss which model performs better on the validation partition and whether the difference is meaningful for the business problem.

Do not assume Ridge will always outperform Linear Regression. Its performance depends on the dataset and regularisation strength.

### 4.11 Visualise Actual vs Predicted Values

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    y_val[:1000],
    val_predictions[:1000],
    alpha=0.5
)

plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs Predicted – Linear Regression")

plt.grid(True)
plt.show()
```

A model with accurate predictions will generally have points closer to the diagonal reference relationship.

### 4.12 Visualise Residuals

```python
residuals = y_val - val_predictions

plt.figure(figsize=(8, 5))

plt.scatter(
    val_predictions[:1000],
    residuals[:1000],
    alpha=0.5
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Target")
plt.ylabel("Residual")
plt.title("Residual Plot – Linear Regression")

plt.grid(True)
plt.show()
```

Look for:

- Residuals scattered around zero.
- Systematic patterns.
- Increasing error spread at higher predicted values.
- Large individual residuals.

These patterns can indicate that the model is missing important relationships or that error behaviour changes across the prediction range.

---

## 5. Real-World Case Study: Delivery Performance Prediction

### Business Scenario

A logistics company wants to improve its delivery planning process.

Operations managers need estimates of a numerical delivery-related outcome to support resource planning, service communication and operational decisions.

The company has historical records containing multiple input features.

### Business Questions

1. Can historical features help estimate the target?
2. How large are the model's typical prediction errors?
3. Does the model perform consistently on unseen records?
4. Does Ridge Regression improve validation performance compared with ordinary Linear Regression?
5. Is the current model sufficiently reliable for the intended operational decision?

### Analytical Approach

| Step | Activity |
|---|---|
| 1 | Understand the target and available features |
| 2 | Use the official training partition |
| 3 | Fit a Linear Regression pipeline |
| 4 | Generate validation predictions |
| 5 | Evaluate MAE, RMSE and R² |
| 6 | Compare with Ridge Regression |
| 7 | Interpret errors and operational limitations |

### Business Interpretation

A model should not be approved simply because it produces a numerical prediction.

Before operational use, the organisation must establish:

- What the target actually represents.
- The unit and any transformation applied to the target.
- Whether the features are available at prediction time.
- Whether the prediction errors are acceptable for the intended decision.
- Whether performance is stable across relevant operational segments.

The model's evaluation metrics are evidence for a decision, not the decision itself.

---

## 6. Hands-On Exercise & Challenge

### Assignment Title: Building and Evaluating a Regression Model

**Problem Statement**

Using the TabReD Delivery ETA dataset, train and evaluate a regression model.

Your task is to implement the workflow and provide a short business interpretation of the results.

### Student Tasks

1. Load the approved numerical, binary and target arrays.
2. Load the official training, validation and test partitions.
3. Create a Linear Regression pipeline.
4. Fit the model using training data only.
5. Generate validation predictions.
6. Calculate MAE, MSE, RMSE and R².
7. Train a Ridge Regression model.
8. Compare both models on the validation partition.
9. Create an actual-versus-predicted chart.
10. Create a residual plot.
11. Write a short recommendation explaining which model performed better and whether the evidence is sufficient for business deployment.

### Expected Output

Students should submit a notebook containing:

- Dataset inspection.
- Feature and target preparation.
- Linear Regression implementation.
- Ridge Regression implementation.
- Evaluation metric results.
- Model comparison table.
- Actual-versus-predicted chart.
- Residual plot.
- Business interpretation.

### Hint / Starter Framework

Use the following model structure:

```python
Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]
)
```

For Ridge Regression, replace the estimator with:

```python
Ridge(alpha=1.0)
```

Use the validation partition for comparison. Keep the test partition separate from model selection.

---

## 7. Knowledge Check

### Conceptual Questions

**1. What is regression used for?**

A. Predicting continuous numerical values  
B. Grouping customers into clusters  
C. Identifying image categories  
D. Sorting records alphabetically

**2. Which variable represents the value we want to predict?**

A. Independent variable  
B. Dependent variable  
C. Index variable  
D. Identifier variable

**3. What does MAE measure?**

A. Average absolute prediction error  
B. Average squared prediction error  
C. Number of features  
D. Model training time

**4. Which metric penalises large prediction errors more strongly because it squares residuals?**

A. MAE  
B. MSE  
C. Feature count  
D. Number of records

**5. What is the purpose of Ridge Regression?**

A. Convert regression into classification  
B. Add L2 regularisation to control coefficient magnitude  
C. Remove all missing values from the target  
D. Automatically select the best dataset

### Answers for Self-Evaluation

1. A – Predicting continuous numerical values.
2. B – Dependent variable.
3. A – Average absolute prediction error.
4. B – MSE.
5. B – Add L2 regularisation to control coefficient magnitude.

---

## 8. Key Takeaways

By the end of Day 4, students should understand that:

- Regression predicts continuous numerical outcomes.
- Linear Regression models relationships between features and a numerical target.
- Multiple Linear Regression uses several input features.
- Residuals help explain prediction errors.
- MAE, MSE, RMSE and R² measure different aspects of regression performance.
- Ridge Regression introduces L2 regularisation.
- Validation performance is more informative than training performance alone.
- Business suitability depends on target meaning, prediction timing and acceptable error—not just a model score.

**Next Session Preview:** Classification – predicting categories and evaluating classification models using confusion matrices and classification metrics.
