# PDC Feature Selection Project

## Team Members
- Aqib Ali (Data Engineering & Environment)
- Ayesha Salahuddin (Fitness & Encoding Logic)
- Syed Hasan Imam (Sequential Baseline Implementation)

## Project Overview
Parallel Feature Selection using competing Add-Remove operators.
Features are represented as Binary Masks and evaluated using GaussianNB.

## Repository Structure
```
pdc-feature-selection/
│
├── milestone-1/
│   └── feature_selection_milestone1.ipynb   # Aqib - data pipeline & evaluator
├── README.md
└── .gitignore
```

## Milestone 1 - Baseline Feature Selection Framework
**Goal:** Establish the environment and sequential performance benchmark.

### Task Breakdown
| Member | Role | Task |
|--------|------|------|
| Aqib Ali | Data Engineering & Environment | Data pipeline, NaiveBayes evaluator ✅ |
| Ayesha Salahuddin | Fitness & Encoding Logic | Binary mask encoding, fitness function |
| Syed Hasan Imam | Sequential Baseline | Sequential search algorithm, visualizations |

### Baseline Results (GaussianNB, All Features)
| Dataset     | Features | Accuracy | Time   |
|-------------|----------|----------|--------|
| MNIST       | 784      | 57.50%   | 0.07s  |
| Madelon     | 500      | 61.54%   | 0.01s  |
| CIFAR-10    | 1024     | 26.50%   | 0.10s  |

## Setup Instructions

### 1. Open the notebook
Open `feature_selection_milestone1.ipynb` in Google Colab

### 2. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Load the pre-split data (get Google Drive link from Aqib)
```python
import numpy as np

X_train = np.load('/content/splits/mnist_X_train.npy')
X_test  = np.load('/content/splits/mnist_X_test.npy')
y_train = np.load('/content/splits/mnist_y_train.npy')
y_test  = np.load('/content/splits/mnist_y_test.npy')
```
Replace `mnist` with `madelon` or `cifar10_gray` for other datasets.

### 4. For Ayesha — Using the NaiveBayes Evaluator
```python
evaluator = NaiveBayesEvaluator()

# Pass a binary mask to evaluate a feature subset
# 1 = keep feature, 0 = drop feature
mask = [1, 0, 1, 1, ...]
accuracy, time_taken = evaluator.evaluate(X_train, X_test, y_train, y_test, feature_mask=mask)
```

### 5. For Hasan — Sequential Baseline Benchmark
The sequential baseline results are in `baseline_results.csv` (shared via Google Drive).
Your sequential search results should be compared against these numbers.

## Dependencies
All pre-installed in Google Colab. No pip installs needed.
- numpy, pandas, sklearn, keras
