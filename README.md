# 🤖 Applied Machine Learning Repository & Study Notes

A comprehensive, hands-on repository covering core machine learning workflows — from statistical exploratory data analysis (EDA) and classic supervised models to advanced ensemble architectures, hyperparameter optimization, unsupervised clustering, and dimensionality reduction.

---

## 📂 Repository Structure

```plaintext
Machine-Learning/
│
├── 📊 Regression/
│   ├── insurance.ipynb          # In-depth EDA, Pearson correlation, Chi-Square hypothesis testing, scaling
│   └── requirements.txt         # Module-specific dependencies
│
├── 🎯 Classification/
│   └── tytanic_modles.ipynb     # Titanic survival benchmark: Logistic Regression, KNN, Naive Bayes, DT, SVM
│
├── ⚡ EnsembleLearning/
│   └── EnsembleLearning.ipynb   # Stacking Classifier, AdaBoost, Gradient Boosting, XGBoost
│
├── ⚙️ model tuning/
│   └── GridSearchCV.ipynb       # Hyperparameter tuning: Exhaustive GridSearchCV vs. RandomizedSearchCV
│
├── 🔍 clustering/
│   └── clustering.ipynb         # K-Means (Elbow method/Inertia) vs. DBSCAN on spherical & non-convex data
│
├── 📉 Dimmension Reduction/
│   └── PCAdimmension.ipynb      # Principal Component Analysis (PCA), feature standardization, 2D latent projection
│
├── .gitignore                   # Ignores virtual envs, checkpoints, and OS artifacts
├── requirements.txt             # Unified project dependencies
└── README.md                    # Master documentation & notes
```

---

## 🧠 Comprehensive Study Notes by Module

### 1. Regression & Statistical Exploratory Data Analysis
📁 **Notebook:** [`Regression/insurance.ipynb`](./Regression/insurance.ipynb)  
🎯 **Dataset:** Medical Cost Personal Dataset (`insurance.csv`)

#### Core Objectives & Techniques
* **Exploratory Data Analysis (EDA):** Analyzed variable distributions (age, BMI, smoker status, region) against healthcare charges using histograms, KDE plots, and box plots.
* **Outlier Analysis & Cleaning:** Identified skewness and heavy tails in financial charges and BMI distributions.
* **Feature Scaling:** Applied `StandardScaler` to bring numeric features (`age`, `bmi`, `children`) to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
* **Statistical Hypothesis Testing:**
  * **Pearson Correlation ($r$):** Measured the linear relationship between continuous predictors and charges ($p$-value evaluation for statistical significance).
  * **Chi-Square Test of Independence ($\chi^2$):** Binned continuous healthcare charges into quartiles using `pd.qcut` and evaluated dependency against categorical factors (e.g., `smoker`, `region`, `sex`) using contingency tables:
    $$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

---

### 2. Classification & Model Benchmarking
📁 **Notebook:** [`Classification/tytanic_modles.ipynb`](./Classification/tytanic_modles.ipynb)  
🎯 **Dataset:** Titanic Passenger Survival

#### Preprocessing Pipeline
* Handled missing values (imputed missing `age` with median, filled `embarked` with mode).
* Encoded categorical features (`sex`, `embarked`) using `LabelEncoder`.
* Standardized features using `StandardScaler` (critical for distance-based and margin-based models).

#### Models Evaluated
| Model | Algorithm Family | Strengths & Considerations |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear / Probabilistic | Baseline classifier, fast, interpretable log-odds coefficients |
| **K-Nearest Neighbors (KNN)** | Instance / Distance-based | Captures local patterns; highly sensitive to scaling and choice of $k$ |
| **Gaussian Naive Bayes** | Probabilistic / Bayes Rule | Assumes feature independence; fast and effective with small datasets |
| **Decision Tree** | Non-parametric Tree | Non-linear splits, interpretable, prone to overfitting without pruning |
| **Support Vector Classifier (SVC)** | Margin Maximization | Uses RBF/linear kernels to find optimal separating hyperplane in high dimensions |

#### Evaluation Metrics
* Evaluated performance using **Accuracy**, **Confusion Matrix**, and precision/recall/F1-score breakdown via `classification_report`.
* Validated generalization stability using `cross_val_score`.

---

### 3. Ensemble Learning
📁 **Notebook:** [`EnsembleLearning/EnsembleLearning.ipynb`](./EnsembleLearning/EnsembleLearning.ipynb)  
🎯 **Objective:** Reduce bias and variance by combining multiple learners.

#### A. Stacking Classifier (`StackingClassifier`)
* **Architecture:**
  * **Base Learners (Level 0):** `DecisionTreeClassifier(random_state=42)` and `SVC()`
  * **Meta-Learner (Level 1):** `LogisticRegression()`
* **Mechanism:** Base models generate out-of-fold predictions via 5-fold cross-validation (`cv=5`). The meta-learner trains on these predictions to make the final decision.

#### B. Boosting Algorithms
* **AdaBoost (`AdaBoostClassifier`):** Sequentially fits weak learners (decision stumps), assigning higher weights to misclassified samples in subsequent iterations.
* **Gradient Boosting (`GradientBoostingClassifier`):** Sequentially minimizes a differentiable loss function by fitting new trees to the pseudo-residuals of previous trees.
* **XGBoost (`XGBClassifier`):** Extreme Gradient Boosting with built-in $L_1$/$L_2$ regularization, tree pruning, cache-aware block structure, and efficient gradient handling.

---

### 4. Hyperparameter Optimization & Model Tuning
📁 **Notebook:** [`model tuning/GridSearchCV.ipynb`](./model tuning/GridSearchCV.ipynb)  
🎯 **Objective:** Systematically discover optimal hyperparameter configurations.

#### Method Comparison
```plaintext
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│          GridSearchCV           │       │       RandomizedSearchCV        │
├─────────────────────────────────┤       ├─────────────────────────────────┤
│ Exhaustive combinatorial search │  vs   │ Randomly samples n_iter points  │
│ Guaranteed best in grid space   │       │ Much faster, budget-friendly    │
│ Computationally expensive       │       │ Excellent for large search grids│
└─────────────────────────────────┘       └─────────────────────────────────┘
```

* **Target Models:** Tuned `SVC` ($C \in \{1, 10, 20, 30\}$, $\text{kernel} \in \{\text{'rbf'}, \text{'linear'}\}$) and `KNeighborsClassifier`.
* **Cross-Validation:** Evaluated across 5 folds (`cv=5`) with `return_train_score=True` to detect overfitting vs. underfitting.
* **Inspection:** Extracted `best_params_`, `best_score_`, and reviewed detailed search trajectories via `cv_results_`.

---

### 5. Unsupervised Learning: Clustering
📁 **Notebook:** [`clustering/clustering.ipynb`](./clustering/clustering.ipynb)  
🎯 **Objective:** Identify natural groupings without ground-truth labels.

#### Algorithms Implemented
1. **K-Means Clustering:**
   * Partitioning method that minimizes within-cluster variance (Inertia):
     $$\text{Inertia} = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$
   * **Elbow Method:** Evaluated inertia across a range of $k \in [1, 10]$ to pinpoint the optimal cluster inflection point.
   * Visualized clusters using `sns.scatterplot(..., hue='cluster', palette='viridis')`.

2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
   * Groups points closely packed together within radius $\epsilon$ (`eps=0.3`) requiring at least `min_samples=5`.
   * Flags sparse outliers as noise label `-1`.
   * **Key Experiment:** Demonstrated why K-Means fails on non-convex datasets (`make_moons`) due to its spherical cluster assumption, whereas DBSCAN effortlessly captures complex, non-linear geometric manifolds.

---

### 6. Dimensionality Reduction (PCA)
📁 **Notebook:** [`Dimmension Reduction/PCAdimmension.ipynb`](./Dimmension Reduction/PCAdimmension.ipynb)  
🎯 **Objective:** Compress high-dimensional feature spaces while preserving maximum variance.

#### Workflow & Mathematical Concepts
1. **Feature Scaling:** Standardized input features with `StandardScaler` so that features with larger numerical scales do not dominate variance calculations.
2. **Eigen-Decomposition:** PCA calculates the covariance matrix, extracting orthogonal eigenvectors (principal components) ordered by eigenvalues (explained variance).
3. **Dimensionality Reduction:** Projected synthetic 5-dimensional data (`n_features=5`) down to a 2D latent space (`n_components=2`).
4. **Visualization:** Plotted the transformed principal components ($PC_1$ vs. $PC_2$) to inspect cluster separability in a low-dimensional plane.

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ installed. Create and activate a virtual environment:

```bash
# Clone the repository
git clone https://github.com/<your-username>/Machine-Learning.git
cd Machine-Learning

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
# or
jupyter lab
```

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Wrangling:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Statistical Analysis:** `scipy.stats`
* **Environment:** `jupyter`, `ipykernel`

---

## 📌 Summary Cheat Sheet

| Domain | Key Algorithms / Tools | Primary Application / Use Case |
| :--- | :--- | :--- |
| **Statistical EDA** | Pearson $r$, Chi-Square ($\chi^2$), `pd.qcut` | Feature dependency, distribution checks & hypothesis validation |
| **Classification** | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM | Discrete target prediction, classification metrics & cross-validation |
| **Ensemble** | Stacking, AdaBoost, Gradient Boosting, XGBoost | Variance reduction, boosting weak learners, competitive predictive power |
| **Model Tuning** | `GridSearchCV`, `RandomizedSearchCV` | Systematic hyperparameter search & cross-validated tuning |
| **Clustering** | K-Means (Elbow method), DBSCAN | Centroid-based vs. density-based unsupervised grouping |
| **Dimensionality Reduction** | Principal Component Analysis (`PCA`) | Variance preservation, feature compression, 2D/3D visualization |

