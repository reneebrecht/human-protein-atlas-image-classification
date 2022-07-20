# Capstone Project: Human Protein Atlas Image Classification

In this project we are building machine learning classification models for automated image analysis. We aim to identify the subcellular locations of proteins inside cultured human cells.  <br>
A protein's location can is connected to its activity within the cell. The location can therefore be a target for pharmaceutical research e.g. while finding a vaccine. Our advanced methods could help researchers to speed up the image analysis process and allow for high troughput imaging.<br>
The data and the idea for the project are from [this](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/overview) kaggle challenge.<br> 
The data are images of cultured human cells. The cells were stained with antibodies and imaged on a fluorescence microscope. The dataset holds 31072 different image stacks. Each stack contains four pictures: from the blue, green, yellow and red channnel of the microscope. The blue, yellow and red channel each contain one spatial marker, that serves as an identifier for one subcellular structure/location. The green channel contains the spatial informaton of the protein of interest, who's location is supposed to be determined.<br>
The protein of interest can be located in more than one location, which makes this task a multi-lable classification problem. 

## Modification of the challenge for our project
The kaggle challenge prompts to use the macro f1 score as metric. The winning model has a macro f1 score 0.59369. Due to time and hardware contrains we decided to modify the challenge to meet our circumstances. Instead of using a pretrained neural network (e.g. ResNet34) like the winner group, we implemented a series of binary classification models for each location, as they are computationaly cheaper. <br>
The second reason for using binary models is that the data at hand are highly imbalanced. To overcome the imbalance we decided to build 28 binary models, one for each location. Our 28 binary models are comparably good at handling imbalanced data as each of them is trained for a single location to predict if the protein of interest is present in this location or not.<br>
 .

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
