# Capstone Project: Human Protein Atlas Image Classification

In this project we are building machine learning classification models for automated image analysis. We aim to identify locations of proteins inside human cells.  <br>
A protein's location can be connected to its activity and is therefore a target for pharmaceutical research e.g. while finding a vaccine. Our advanced methods can help researchers save time and money.<br>
The data and the idea of the project is from a [challenge](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/overview). We've got 31072 different packages with four pictures each. The first of the four pictures shows the fluorescent protein and the other three show markers which are used to identify the location.<br>
The pictures of one package can include one or more locations. 

## Modification of the challenge for our project
The winner model has a score 0.59369. By building a model from scratch we won't hardly improve this result. So we decide to modify the challenge. <br>
The data is highly imbalanced. The two locations most often find in the pictures are on 56,3% of the pictures. <br>
To overcome the imbalance we decide on building 28 binary models. Each of these models is trained for one possible location and predict if there is this one location on the picture ore not.

## Requirements:

- pyenv with Python: 3.9.4

### Setup
To run our model the following setups. 
```
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Stakeholder
Our Stakeholder is the same stakeholder than in the challenge. We want to help a company which build microscopes to improve there smart-microscopes. 
## Value 
Proteins are important for all kind of processes in a cell. For example to create something like a vaccine the researches needs to know how to get a protein on the right location in a cell, so it can protect us. <br>
In the moment modern microscopes can produce a high number of pictures of the proteins in a cell but the determination of the location are still made by experts who needs a lot of training and time. This time can cost a lot of money and in case of pharamceutical research also the health and lives of humans and animals. <br>
To implement models who can make the work of these experts easier and faster is therefore highly necessary. 
## Baseline model
In our team are three studied biologists how can identify the locations of proteins. So we decided their results are our baseline model. This model can represent the way the determination is still made. We know from it that determine 164 pictures can take 3 hours and archieves a macro f1 score of 0.32. This is the model we want to improve. 
## The way to our model
In the notebooks in this repository we documented our work. If you comprehend our work, please feel free to read:
 - [First: EDA](https://github.com/reneebrecht/human-protein-atlas-image-classification/blob/main/notebooks/EDA.ipynb)

If you want to run the notebooks you have to save the data from the [challenge](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/overview) in a new created folder `train` in [data](https://github.com/reneebrecht/human-protein-atlas-image-classification/tree/main/data). You also need the empty folders `embeddings_train` and `images_train_tfrec` in this directory. 
## The results
The best models we got for all of the location are saved in the folder [models](https://github.com/reneebrecht/human-protein-atlas-image-classification/tree/main/models).
We archived a macro f1-score of 0.67.
