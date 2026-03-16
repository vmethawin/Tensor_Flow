# House Prices - Advanced Regression Techniques

A machine learning ensemble model for predicting house prices on the Kaggle House Prices dataset.

## Project Overview

This project uses a **5-model ensemble** combining neural networks and random forests to predict house prices:

- **3 Neural Networks** (different architectures)
  - Model 1: 64 → 64 ReLU layers
  - Model 2: 64 ReLU layer
  - Model 5: 100 → 64 ReLU layers

- **2 Random Forests** (TensorFlow Decision Forests)
  - Model 3: 1000 trees (seed: 1234)
  - Model 4: 1000 trees (seed: 4567)

All predictions are averaged for the final result.

## Setup

### Requirements
- Python 3.11+
- TensorFlow & Keras
- scikit-learn
- pandas, numpy, matplotlib

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow tensorflow-decision-forests scikit-learn pandas numpy matplotlib pydot graphviz
```

### MacOS M1/M2 Note
- Uses legacy Keras optimizer for better performance
- Graphviz required for model visualization

```bash
brew install graphviz
```

## Usage

1. **Run the notebook**:
   ```bash
   jupyter notebook Simple\ model.ipynb
   ```

2. **Execute cells in order**:
   - Setup & imports
   - Load & explore data
   - Preprocess
   - Build & train models
   - Evaluate
   - Generate submission

3. **Output**:
   - `submission.csv` - Final predictions for Kaggle

## Project Structure

```
.
├── Simple model.ipynb          # Main notebook
├── submission.csv              # Kaggle submission
├── house-prices-*/            # Training & test data
├── venv/                       # Virtual environment
└── README.md                   # This file
```

## Model Performance

- **MAE** (Mean Absolute Error): ~$XX,XXX
- **RMSE** (Root Mean Squared Error): ~$XX,XXX
- **R² Score**: 0.XXXX

(Update after training)

## Kaggle Competition

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Author

Data Science Project - 2026
