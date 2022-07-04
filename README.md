# Capstone Project: Human Protein Atlas Image Classification

This project is about identifying the whereabouts of a specific proteins in human cells. <br>
In the original [challenge](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/overview) the exercise was to build a model which can identify the locations of proteins in a cell. Therefor the given data for Training the model are pictures from marked proteins in cells. Each protein positions is described by four pictures. <br>
The challenge had the purpose to help building a "smart-microscope" which can fasten the determination of proteins location in comparison of a trained human being.<br>
The location is needed in the field of biomedicine and modern microscopes can produce too high number of pictures for the experts to analyze manually. That is why such models are needed. <br>
## Modification of the challenge for our project
The winner model has a score 0.59369. By building a model from scratch we won't hardly improve this result. So we decide to use this results and improve it. <br>
The data is highly imbalanced. The two most frequent locations in the pictures are Nucleoplasm and Cytosol. 56,7% of the pictures shows at least one of those two locations. <br>
So we decide on building two models. One model which only is trained on finding the best two most frequently locations and another which is trained to decide is a protein in a specific location or not. We want to make predication with a score higher than 0.7.<br>
Another idea is, that we use only the pictures of the cells where trained experts can identify the location correctly. The other pictures do not seem good enough to identify the locations and are so not improving but aggravate the model. We will try to get the results from the experts. 

## Requirements:

- pyenv with Python: 3.9.4

### Setup
To run our model the following setups. 
```
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```

The `requirements.txt` file contains the libraries needed for deployment.

## Stakeholder
Our Stakeholder is the same stakeholder than in the challenge. We want to help a company which build microscopes to improve there smart-microscopes. 
## Value 
Smart-microscopes which can identify the locations of proteins much faster than humans are needed because newer microscopes can make pictures of the cells much faster than the available trained humans can analyze. The information is important for research in medicine. 

## Baseline model
In our team are three studied biologists how can identify the locations of proteins. So we decided their results are our first model.

## Metric
In the challenge the requested metric is the macro F1 score. We decided to use also the macro F1 score for our baseline model and for our multilabel models. For the binary models we will use the F1 score. 