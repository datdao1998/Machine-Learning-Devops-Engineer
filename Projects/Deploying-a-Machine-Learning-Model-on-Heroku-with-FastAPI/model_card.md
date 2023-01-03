# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
## Model Details
It is a Random forest classifier for prediction using default configuration for training with scikit-learn.

n_splits=10, n_estimators=100

## Intended Use
This model should be used to predict the category of the salary of a person based on it's financials attributes.

## Metrics
The overall model performance was evaluated using Fbeta score, and Precision-Recall Score

Script use to classify : **starter/classification_metrics.py**

Precision Score :  0.9453991130820399

Recall Score :  0.9088729016786571

FBeta Score :  0.9267762532264638

## Training Data
UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income)
The original data set has 32561 rows and 15 columns
80% of the data is used to train model

## Evaluation Data
20 % data is used to evaluated

## Ethical Considerations
The model reflects the social composition of that time in the United States.

## Caveats and Recommendations
This model was built as a Proof of Concept, not for production
The data is outdated, so to be used it should be updated
