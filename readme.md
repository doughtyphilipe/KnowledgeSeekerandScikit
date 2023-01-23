# Knowledge Seeker

The Knowledge Seeker Method, originally described in [Knowledge Seeker - Ontology Modelling for Information Search and Management](https://link.springer.com/book/10.1007/978-3-642-17916-7), was applied in German Industry Norms ([DIN](https://en.wikipedia.org/wiki/Deutsches_Institut_f%C3%BCr_Normung)). This thesis debates the usage, the hurdles and algorithm behind this method. The results can be found in the knowledge_seeker.ipynb file.


## Usage

```python
import fitz

## import stamp 
tke = open("./stamps/LogoTKE_reveal.png", "rb").read()

## define coordinates
mage_rectangle1 = fitz.Rect(1854,1524,2154,1564)

## insert image (stamp)
first_page.insert_image(image_rectangle1, stream=tke)

## save outputpdf

file_handle.save(output)

```

# Scikit

The Scikit methods for text classification are broadly debated [here](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). We applied and disscused the diferences between these methods and the Knowledge Seeker method. The results can be found in the norm_classifier.ipynb file.

## Installation

Scikit can be installed as follows.

```bash
pip install -U scikit-learn
```

Importing in a python file.

```python
import sklearn
```


