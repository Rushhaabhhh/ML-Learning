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
   - Formula:
     $$
     \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
     $$
   - **Definition**: The proportion of correct predictions (both true positives and true negatives) out of all predictions.

2. **Precision**:
   - Formula:
     $$
     \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
     $$
   - **Definition**: The proportion of positive predictions that are actually correct.

3. **Recall**:
   - Formula:
     $$
     \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
     $$
   - **Definition**: The proportion of actual positives that are correctly identified by the model.

4. **F1-Score**:
   - Formula:
     $$
     \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
     $$
   - **Definition**: The harmonic mean of precision and recall, providing a balance between the two.

5. **F-beta Score**:
   - Formula:
     $$
     F_{\beta} = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}
     $$
   - **Definition**: A generalization of the F1-score that allows you to control the balance between precision and recall using the parameter \( \beta \).

6. **Confusion Matrix**:
   - A **Confusion Matrix** is a table that describes the performance of a classification model by comparing the actual and predicted values. It consists of:
     - **True Positives (TP)**: Correctly predicted positive cases.
     - **True Negatives (TN)**: Correctly predicted negative cases.
     - **False Positives (FP)**: Incorrectly predicted positive cases.
     - **False Negatives (FN)**: Incorrectly predicted negative cases.

---

#### Regression Metrics

1. **Mean Squared Error (MSE)**:
   - Formula:
     $$
     \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
     $$
   - **Definition**: The average of the squared differences between the actual and predicted values.

2. **Root Mean Squared Error (RMSE)**:
   - Formula:
     $$
     \text{RMSE} = \sqrt{\text{MSE}}
     $$
   - **Definition**: The square root of MSE, which gives an error metric in the same units as the target variable.

3. **Mean Absolute Error (MAE)**:
   - Formula:
     $$
     \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
     $$
   - **Definition**: The average of the absolute differences between the actual and predicted values.

4. **R² Score**:
   - Formula:
     $$
     R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
     $$
   - **Definition**: Measures how well the model explains the variation in the target variable. A value closer to 1 indicates a better model fit.

5. **Adjusted R² Score**:
   - Formula:
     $$
     R_{\text{adj}}^2 = 1 - \left(1 - R^2\right) \times \frac{n-1}{n-p-1}
     $$
   - **Definition**: A modified version of R² that adjusts for the number of predictors in the model, preventing overfitting.

---

### Linear Regression
Linear regression attempts to model the relationship between two variables by fitting a linear equation to the observed data.

#### Formula for Linear Regression:
The equation of a straight line is:
$$
y = mx + b
$$
Where:
- \( y \) is the target variable,
- \( x \) is the feature variable,
- \( m \) is the slope (coefficient),
- \( b \) is the y-intercept.

---

### Logistic Regression
Logistic regression is used for binary classification tasks. The output is between 0 and 1, representing the probability of a class.

#### Formula for Logistic Regression:
The logistic function (sigmoid function) is:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
Where:
- \( z = \theta^T x \), the linear combination of the input features.
