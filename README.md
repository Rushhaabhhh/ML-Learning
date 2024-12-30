# Machine Learning Concepts

## 📚 Fundamental Concepts

### Supervised Learning
Supervised learning is a type of machine learning where the model is trained on labeled data, meaning that each training example is paired with an output label. The goal is to learn a mapping from inputs to outputs.

#### Types of Supervised Learning:
1. **Classification**: The task of predicting a discrete label (e.g., spam or not spam).
   - Example: Predicting whether an email is spam or not.
   
2. **Regression**: The task of predicting a continuous value (e.g., predicting house prices).
   - Example: Predicting the price of a car based on features like make, model, and year.

---

### Model Evaluation Metrics

#### Classification Metrics
When evaluating classification models, several metrics help measure their performance:

1. **Accuracy**:
   - **Formula**: 
     ```
     Accuracy = (TP + TN) / (TP + TN + FP + FN)
     ```
   - **Definition**: The proportion of correct predictions (both true positives and true negatives) out of all predictions.

2. **Precision**:
   - **Formula**: 
     ```
     Precision = TP / (TP + FP)
     ```
   - **Definition**: The proportion of positive predictions that are actually correct.

3. **Recall**:
   - **Formula**: 
     ```
     Recall = TP / (TP + FN)
     ```
   - **Definition**: The proportion of actual positives that are correctly identified by the model.

4. **F1-Score**:
   - **Formula**: 
     ```
     F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
     ```
   - **Definition**: The harmonic mean of precision and recall, providing a balance between the two.

5. **F-beta Score**:
   - **Formula**: 
     ```
     F-beta = (1 + β²) * (Precision * Recall) / ((β² * Precision) + Recall)
     ```
   - **Definition**: A generalization of the F1-score that allows you to control the balance between precision and recall using the parameter β.

6. **Confusion Matrix**:
   - A table that describes the performance of a classification model by comparing the actual and predicted values:
     - **True Positives (TP)**: Correctly predicted positive cases.
     - **True Negatives (TN)**: Correctly predicted negative cases.
     - **False Positives (FP)**: Incorrectly predicted positive cases.
     - **False Negatives (FN)**: Incorrectly predicted negative cases.

   **Confusion Matrix Structure**:
   ```
   Predicted \ Actual | Positive | Negative
   -------------------|-----------|-----------
   Positive           | TP        | FP
   Negative           | FN        | TN
   ```

---

#### Regression Metrics

1. **Mean Squared Error (MSE)**:
   - **Formula**: 
     ```
     MSE = (1/n) * Σ(y_i - ŷ_i)²
     ```
   - **Definition**: The average of the squared differences between the actual and predicted values.

2. **Root Mean Squared Error (RMSE)**:
   - **Formula**: 
     ```
     RMSE = √(MSE)
     ```
   - **Definition**: The square root of MSE, which gives an error metric in the same units as the target variable.

3. **Mean Absolute Error (MAE)**:
   - **Formula**: 
     ```
     MAE = (1/n) * Σ|y_i - ŷ_i|
     ```
   - **Definition**: The average of the absolute differences between the actual and predicted values.

4. **R² Score**:
   - **Formula**: 
     ```
     R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
     ```
   - **Definition**: Measures how well the model explains the variation in the target variable. A value closer to 1 indicates a better model fit.

5. **Adjusted R² Score**:
   - **Formula**: 
     ```
     R²_adj = 1 - ((1 - R²) * (n-1) / (n-p-1))
     ```
   - **Definition**: A modified version of R² that adjusts for the number of predictors in the model, preventing overfitting.

---

### Linear Regression
Linear regression attempts to model the relationship between two variables by fitting a linear equation to the observed data.

#### Formula for Linear Regression:
```
y = mx + b
```
Where:
- `y` is the target variable
- `x` is the feature variable
- `m` is the slope (coefficient)
- `b` is the y-intercept

**Multiple Linear Regression**:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
Where:
- `β₀` is the intercept
- `β₁, β₂, ..., βₙ` are the coefficients
- `x₁, x₂, ..., xₙ` are the independent variables
- `ε` is the error term

---

### Logistic Regression
Logistic regression is used for binary classification tasks. The output is between 0 and 1, representing the probability of a class.

#### Formula for Logistic Regression:
1. **Sigmoid Function**:
   ```
   σ(z) = 1 / (1 + e^(-z))
   ```

2. **Logistic Regression Equation**:
   ```
   P(y=1) = σ(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)
   ```
Where:
- `z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`
- `σ` is the sigmoid function
- `β₀` is the intercept
- `β₁, β₂, ..., βₙ` are the coefficients
- `x₁, x₂, ..., xₙ` are the independent variables

---
