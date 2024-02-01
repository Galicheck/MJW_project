# Predicting Wine Quality with SGD Classifier

This project aims to predict the quality of wine using Stochastic Gradient 
Descent (SGD) Classifier.

## About the dataset

The dataset contains information on the physicochemical properties and quality ratings 
of red and white variants of the Portuguese “Vinho Verde” wine.
The goal of the dataset is to explore the relationship between these physicochemical
attributes and the sensory-based quality score assigned to each wine. 
The data can be utilized for both classification and regression tasks, 
offering insights into predicting wine quality based on its chemical properties.

It's important to note that the classes in the dataset are ordered,
and there is an imbalance among them. Specifically, there are more instances
of normal wines compared to excellent or poor ones.

Source: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

This dataframe contains the following columns (input variables are based on physicochemical tests):

| Variable names                   |
|----------------------------------|
| fixed acidity                    |
| volatile acidity                 |
| citric acid                      |
| residual sugar                   |
| chlorides                        |
| free sulfur dioxide              |
| total sulfur dioxide             |
| density                          |
| pH                               |
| sulphates                        |
| alcohol                          |
| quality (score between 0 and 10) |

The output variable is wine quality, which is based on sensory data.

## How to run this project

### Setting up virtual environment and installing requirements

After cloning the repository set up virtual environment and install the necessary requirements:
```
python -m venv venv 
venv\Scripts\activate
pip install -r requirements.txt
```

### Running the project using DVC

This project uses DVC (Data Version Control) to run data processing pipelines.
If you have access to remote SSH storage, you can simply pull the data:
```
dvc pull
```
Then reproduce the stages by running:
```
dvc repro -f
```

### Running the Project Without Access to Remote Storage

If you don't have access to remote storage, you can run the project by downloading the original data `winequalityN.csv`
from the Kaggle website: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset. Then add downloaded file to data
folder and run the following commands:
```
mkdir plots
dvc repro
```

### Seeing the results

After reproducing stages you can see resulting confusion matrices. Accuracies of classifiers, for red and white wine
separately, are printed in terminal.