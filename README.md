#  Student Performance Analysis: Unsupervised Clustering vs. Supervised Neural Network

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F79A3E?style=flat&logo=scikit-learn)

An end-to-end Machine Learning research project exploring student performance patterns. This repository compares **unsupervised clustering (K-Means)** on study habits with a **supervised Deep Neural Network (PyTorch)** trained to predict performance groups without data leakage.

---

##  Research Question

> **"How well do natural student clusters based on study habits align with academic performance groups, and how accurately can a supervised Neural Network predict these groups?"**

---

##  Key Methodology & Improvements

Based on rigorous peer review, the analysis workflow addresses data leakage and severe class imbalance through the following fixes:

1. **Quantile-Based Target Balancing:** Replaced fixed threshold cuts with **quantile splits** (33% tertiles via `pd.qcut`), resolving severe class imbalance and creating equal representation across `Low`, `Medium`, and `High` academic performance groups (~33.3% per group).
2. **Data Leakage Elimination:** `Exam_Score` was strictly isolated from input features. Feature scaling (`StandardScaler`) is fitted **only on the training set** and applied to Validation/Test sets.
3. **Stratified Split:** Data split into **Train / Validation / Test (70% / 15% / 15%)** maintaining class balance across all subsets using `stratify=y`.
4. **Validation-Guided Model Checkpointing:** Tracked **Validation Macro F1-Score** per epoch to save the optimal weights (`best_nn_model.pth`) and prevent overfitting.

---

##  Features & Dataset

* **Dataset Source:** Kaggle Student Performance Factors (~6,600 records cleaned with `dropna()`).
* **Features ($X$):** `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`.
* **Target ($y$):** `Performance_Group` (0: Low, 1: Medium, 2: High).

---

##  Models & Architectures

1. **Unsupervised (K-Means):** Evaluated $k \in [2, 6]$ using Inertia (Elbow method) and Silhouette score to justify $k=3$. Measured alignment with ground truth targets via **Adjusted Rand Index (ARI)**.
2. **Supervised (PyTorch MLP):** Compared three architectures with ReLU activations, Dropout (0.2), CrossEntropyLoss, and Adam optimizer ($\alpha=0.001$):
   * **Arch 1:** `6 → 8 → 3`
   * **Arch 2:** `6 → 16 → 8 → 3`
   * **Arch 3:** `6 → 32 → 16 → 3`

---

##  Final Test Metrics & Results

### K-Means Clustering Alignment
* **Silhouette Score ($k=3$):** ~0.12–0.30
* **Adjusted Rand Index (ARI):** ~0.00 (Demonstrates that natural student lifestyle clusters do not map directly to strict grade tiers without supervised guidance)

### Neural Network Test Performance (Best Architecture)

| Metric | Score |
| :--- | :---: |
| **Test Accuracy** | **0.7524** |
| **Macro F1-Score** | **0.7430** |



---

## 🚀 How to Run

1. Clone repository:
   ```bash
   git clone [https://github.com/tomikakenzhetaeva23-star/student-perfomance-patterns.git](https://github.com/tomikakenzhetaeva23-star/student-perfomance-patterns.git)
