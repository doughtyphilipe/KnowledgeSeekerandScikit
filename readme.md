
# Automatic identification and provision of user-specific knowledge (KnowledgeSeeker and Scikit)
This is a Google Colab application that uses two different approaches (KnowledgeSeeker and Scikit) to classify German industry norms (DIN). The code was the base for a bachelor thesis called "Automatic identification and provision of user-specific knowledge".

## Knowledge Seeker

The Knowledge Seeker Method, originally described in [Knowledge Seeker - Ontology Modelling for Information Search and Management](https://link.springer.com/book/10.1007/978-3-642-17916-7), was applied in German Industry Norms ([DIN](https://en.wikipedia.org/wiki/Deutsches_Institut_f%C3%BCr_Normung)). This thesis debates the usage, the hurdles and algorithm behind this method. The results can be found in the knowledge_seeker.ipynb file.

The following picture describes a first analysis, right after training our model with springer link books. In this case the domain area was "Konstruktion" (Mechanical Engineering Design).
![knowledgeseeker](https://github.com/doughtyphilipe/KnowledgeSeekerandScikit/blob/main/knowledgeseeker.jpg)


## Scikit

The Scikit methods for text classification are broadly debated [here](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). We applied and disscusedthe diferences between these methods and the Knowledge Seeker method. The results can be found in the norm_classifier.ipynb file.

The following picture describes a general text pre-processing method that we applied in this thesis.
![scikit](https://github.com/doughtyphilipe/KnowledgeSeekerandScikit/blob/main/preprocessing.PNG)

The following picture describes one of the final results, a classified DIN Norm.
![scikit](https://github.com/doughtyphilipe/KnowledgeSeekerandScikit/blob/main/xmlexample.jpg)

## Features
Classification of German industry norms (DIN) using two approaches
Implementation of KnowledgeSeeker algorithm
Use of Scikit-learn library for classification

## Installation
Go to the Google Colab website
Click on "File" and select "Upload notebook"
Upload the KnowledgeSeeker and Scikit.ipynb file from this repository
Follow the instructions in the notebook to run the code

for Scikit:
```bash
pip install -U scikit-learn
```

Importing in a python file.

```python
import sklearn
```

## Usage
Open the KnowledgeSeeker and Scikit.ipynb file in Google Colab
Follow the instructions in the notebook to load the DIN dataset and preprocess it for classification
Choose which approach to use for classification: KnowledgeSeeker or Scikit-learn
Train the classification model using the selected approach
Test the model's performance on a validation dataset
Use the model to classify new DINs


