This project demonstrates a **comparison between multiple Machine Learning models** for text classification using **Scikit-Learn**.  
The goal is to identify which algorithm performs best in terms of **Accuracy, Precision, Recall, and F1-score**.

---

## 📂 Project Overview

In this project, several ML algorithms are applied and compared on a text dataset:

- **SVM (Linear and RBF kernels)**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**

Each model was evaluated using **cross-validation** to ensure fair comparison between models.

---

## ⚙️ Technologies Used

- Python 🐍  
- Scikit-Learn  
- NumPy  
- Pandas  
- Jupyter Notebook  

---

## 🧪 Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1-score |
|--------|-----------|------------|----------|-----------|
| SVM (Linear) | 0.966 | 0.959 | 0.959 | 0.959 |
| SVM (RBF) | 0.964 | 0.948 | 0.968 | 0.957 |
| **Logistic Regression** ✅ | **0.979** | **0.971** | **0.979** | **0.975** |
| KNN | 0.825 | 0.804 | 0.863 | 0.811 |
| Decision Tree | 0.790 | 0.833 | 0.650 | 0.667 |

> 🔹 The best-performing model was **Logistic Regression**, showing strong and balanced performance across all metrics.

---

## 🧠 Explanation

The project workflow:

1. **Text Vectorization** using `CountVectorizer` to convert text data into numerical format.  
2. **Pipeline Creation** to combine vectorization and the classifier into one clean process.  
3. **Cross-validation** (`cross_val_score` / `cross_validate`) to evaluate each model reliably.  
4. **Model Comparison** using evaluation metrics (Accuracy, Precision, Recall, F1-score).

Each model was trained and validated using the same data split and evaluation procedure to ensure a fair comparison.

---

## 🧩 Example Code Snippet

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Example with SVM (Linear)
svm = SVC(kernel='linear')

pipe_svm = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('svm', svm)
])

scores = cross_validate(
    pipe_svm,
    corpus, y,
    cv=5,
    scoring={
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    }
)

print("Accuracy:", scores['test_accuracy'].mean())
print("Precision:", scores['test_precision'].mean())
print("Recall:", scores['test_recall'].mean())
print("F1-score:", scores['test_f1'].mean())
