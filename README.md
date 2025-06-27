# Logistic_Regression_Classifier
# 🧪 Logistic Regression Binary Classifier

A binary classification project using Logistic Regression on the Breast Cancer Wisconsin dataset. This project includes data preprocessing, model training, evaluation, and visual analysis using Python.

---

## 📁 Files Included

- `logistic_regression_classifier.py`: Main Python script to run the model
- `preprocessed_data.csv`: Cleaned and standardized dataset
- `confusion_matrix.png`: Confusion matrix plot
- `roc_curve.png`: ROC-AUC curve plot
- `sigmoid_function.png`: Visualization of the sigmoid function

---

## 🚀 How to Run the Code

1. **Install dependencies** (if not already installed):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

---

## 💬** Logistic Regression Interview Questions**

This section includes commonly asked interview questions on logistic regression with concise and clear answers to help you prepare.

---

### 1. **How does logistic regression differ from linear regression?**
- **Linear regression** predicts continuous values (e.g., price, age).
- **Logistic regression** predicts probabilities for classification problems (e.g., spam or not spam).
- Logistic regression uses the **sigmoid function** to convert linear output into a probability between 0 and 1.

---

### 2. **What is the sigmoid function?**
- The **sigmoid function** maps any real number to a value between 0 and 1.
- It is defined as:

  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]

- In logistic regression, it’s used to convert linear outputs to probabilities.

---

### 3. **What is precision vs recall?**
- **Precision** = TP / (TP + FP) → How many predicted positives are actually positive.
- **Recall** = TP / (TP + FN) → How many actual positives were correctly predicted.
- Use **precision** when false positives are costly, and **recall** when false negatives are costly.

---

### 4. **What is the ROC-AUC curve?**
- **ROC (Receiver Operating Characteristic)** plots the **True Positive Rate (Recall)** against the **False Positive Rate** at various thresholds.
- **AUC (Area Under Curve)** indicates how well the model distinguishes between classes:
  - AUC = 1 → perfect classifier
  - AUC = 0.5 → random guessing

---

### 5. **What is the confusion matrix?**
A 2x2 table showing prediction results:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | True Positive (TP)   | False Negative (FN) |
| Actual Negative | False Positive (FP)  | True Negative (TN)  |

---

### 6. **What happens if classes are imbalanced?**
- The model may be biased toward the majority class.
- Accuracy becomes a misleading metric.
- Solutions:
  - Use metrics like precision, recall, F1-score
  - Use techniques like **SMOTE**, **undersampling**, or **class weighting**

---

### 7. **How do you choose the threshold?**
- Default threshold = 0.5, but not always optimal.
- Use **ROC curve**, **precision-recall trade-offs**, or **domain requirements** to tune it.
- You can plot metrics vs. thresholds to choose the best cutoff point.

---

### 8. **Can logistic regression be used for multi-class problems?**
- Yes, using:
  - **One-vs-Rest (OvR)** strategy (default in scikit-learn)
  - **Softmax / Multinomial Logistic Regression** for directly predicting multi-class probabilities

---
