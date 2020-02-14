_Project done in the Machine Learning course (CS-433) at EPFL_

# Higgs boson classifier
In this project we try to determine if a given event's signature was the result of a Higgs boson (signal) or some other particle (background) thanks to a vector representing the decay signature of an event. The data set is provided by the EPFL on the website AIcrowd and separated into two parts: a test and a training set. Therefore, the goal is to choose an appropriate machine learning model, such as logistic regression, least squares or ridge regression and train it on the training set, in order to ultimately get the best results on the test set.

## Processing
All the processing, feature engineering, feature expansion is explained in the pdf file

## Files
- `implementations.py` contains the 6 methods we were asked to implement.
- `helpers.py` the other methods we need to run the ML algorithm
- `proj1_helpers.py` the helper methods we were given for this project
- `README.md` this README file

## Conventions
For vectors of any size N, we always assume that the numpy shape is (N,1)
`run.py` assumes the data `test.csv` and `train.csv` is in folder `data/`

## Run code
You need python 3 and numpy, for example if you have pip:
```
pip install -U numpy
python run.py
```
This will create a predictions-run.csv file which are the test predictions.
