# Neural Network Classifier for CSC4025Z AI

This project demonstrates the development and training of neural network classifiers using PyTorch. The classifiers are trained on datasets from the UCI Machine Learning Repository and Kaggle.

## Table of Contents
- [Installation](#installation)
- [Datasets](#datasets)
- [Data Processing](#data-processing)
- [Model Development](#model-development)
- [Training](#training)
- [Evaluation](#evaluation)
- [Grid Search](#grid-search)

## Installation

To run this project, you need to have Python 3.10 and the following packages installed:

- pandas
- scikit-learn
- torch
- matplotlib
- ucimlrepo

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Datasets

The project supports two datasets:
- Fetal Health (Kaggle)
- Steel Plates Faults (UCI Machine Learning Repository) [obsolete]

You can configure the dataset to be used by setting the `DATASET` variable in the notebook.

## Data Processing

The data processing steps include:
1. Loading the data
2. Splitting the data into training, validation, and test sets
3. Normalizing the data
4. Converting the data to PyTorch tensors

## Model Development

The project includes implementations of three neural network models:
- Multi-Layer Perceptron (MLP)
- Deep Multi-Layer Perceptron (DeepMLP)
- Convolutional Neural Network (CNN)

## Training

The training loop includes:
- Setting hyperparameters
- Training the model with early stopping
- Logging training and validation losses

## Evaluation

The evaluation includes:
- Plotting training and validation losses
- Computing classification metrics such as accuracy, precision, recall, and F1 score

## Grid Search

The project includes a grid search implementation to find the best hyperparameters for the models.

## Example Usage

To train a model, set the `SHOW_EXAMPLE` variable to `True` and run the notebook. The training loop will print the training and validation losses and plot the loss curves.

```python
SHOW_EXAMPLE = True
```

For more details, refer to the notebook `classifier.ipynb`.


## Team

2023 Simbarashe Mawere, Sam Frost, Fabio O'Paulo Ryan
