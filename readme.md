
# Titanic Survival Prediction

## Introduction

This project analyzes the Titanic dataset to predict passenger survival using machine learning models. The project includes data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Ndrake337/Kagle_Titanic.git
    cd Kagle_Titanic
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the analysis and prediction:

1. Ensure you have the Titanic dataset (`train.csv` and `test.csv`) in the project directory.
2. Open and run the Jupyter notebook:
    ```sh
    jupyter notebook Titanic.ipynb
    ```

## Features

- Data loading and preprocessing
- Feature engineering
- Model training using RandomForestClassifier and LogisticRegression
- Model evaluation using cross-validation
- Error analysis

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn

## Configuration

Ensure the following configurations in the notebook:

- File paths for `train.csv` and `test.csv`.
- Random state and other parameters for reproducibility.

## Documentation

### Data Loading

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

### Feature Engineering

Converting categorical 'Sex' feature to binary:

```python
def binarySex(valor):
    if valor == 'female':
        return 1
    else:
        return 0

train['Sex_binario'] = train['Sex'].map(binarySex)
test['Sex_binario'] = test['Sex'].map(binarySex)
```

### Model Training and Evaluation

Using Repeated K-Fold Cross-Validation with RandomForestClassifier:

```python
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier

resultados = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=10)

for linhas_treino, linhas_valid in kf.split(X):
    X_treino, X_valid = X.iloc[linhas_treino], X.iloc[linhas_valid]
    y_treino, y_valid = y.iloc[linhas_treino], y.iloc[linhas_valid]
    
    modelo = RandomForestClassifier(n_estimators = 100, n_jobs=-1, random_state=0)
    modelo.fit(X_treino, y_treino)
    
    p = modelo.predict(X_valid)
    acc = np.mean(y_valid == p)
    resultados.append(acc)
```

## Examples

Analyzing errors in predictions:

```python
X_valid_check = train.iloc[linhas_valid].copy()
X_valid_check['p'] = p
erros = X_valid_check[X_valid_check['Survived'] != X_valid_check['p']]
```

## Troubleshooting

- Ensure all dependencies are installed.
- Verify the paths to the dataset files are correct.
- Check for missing values and handle them appropriately before training.

## Contributors

- Your Name - [@Ndrake337](https://github.com/Ndrake337)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
