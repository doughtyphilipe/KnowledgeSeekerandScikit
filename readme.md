# Automatic identification and provision of user-specific knowledge (KnowledgeSeeker and Scikit)
This is a Google Colab application that uses two different approaches (KnowledgeSeeker and Scikit) to classify German industry norms (DIN). The code was the base for a bachelor thesis called "Automatic identification and provision of user-specific knowledge".

![knowledgeseeker](https://user-images.githubusercontent.com/123315352/224511538-06e46c40-334c-4c7d-ab49-cf1835b1421a.jpg)

## NLP in general

First we set our data directy and labels from the folder Books_Classification. This folder was in my Google Drive account and contains 44 books from 8 different knowledge areas related to mechanical engineering.

```python
BASE_DIR = '/content/drive/MyDrive/ColabNotebooks/Norm_Classifier/Book_Classfication'
LABELS = []
for label in os.listdir(BASE_DIR):
  LABELS.append(label)
``` 

We then load generated docs from content/sample_data/Files/data.pkl and create Pandas dataframe.

```python
docs = pd.read_pickle('/content/drive/MyDrive/ColabNotebooks/Norm_Classifier/Files/data.pkl')
```

This dataframe give us a general idea of what kind of data we are working with.

The following picture describes a general text pre-processing method that we applied in this thesis. We need to filter out "useless" words and focus on the meaningful tokens for our analysis.
![preprocessing](https://user-images.githubusercontent.com/123315352/224511545-698d4984-5496-44c9-b99f-516a5255b5ad.PNG)

For preoprocessing, we will use the following modules: [NLTK](https://www.nltk.org/) , [SnowballStemmer](https://www.nltk.org/_modules/nltk/stem/snowball.html) , [HannoverTagger](https://textmining.wp.hs-hannover.de/Preprocessing.html) and the [PyChant](https://pyenchant.github.io/pyenchant/tutorial.html) Dictionary.

The code that filters out the unecessary words is the following:

```python
def tokenize_and_lemma(text):

    tokens = nltk.word_tokenize(text)
      
    filtered_tokens = []

    lemmas = []
    
    stemmer = SnowballStemmer("german")

    tagger = ht.HanoverTagger('morphmodel_ger.pgz')

    specialchar = ['.',',',';',':','"','-','(',')','[',']','{','}','/','//','>','<','=','+','-','#','@','::','+//','...']

    deutsch = enchant.Dict("de_DE_frami")
    
    for token in tokens:
      if any(map(str.isdigit, token)) == False:        
        if any(map(lambda x: [x for x in specialchar],token)):
          if token not in stop_words:
            if '-' in token:                  
              token = token.replace('-','')
            elif '.' in token:                  
              token = token.replace('.','')
            if len(token) > 2:
              if deutsch.check(token) == True:
                lemma = [lemma for (word,lemma,pos) in tagger.tag_sent(token.split())]                
                if lemma != ['--']:     
                  lemma_low = (map(lambda x: x.lower(), lemma))           
                  lemmas.append(' '.join(lemma_low))                  
    return lemmas
```

An example of the code working would be:

```bash
string = 'Diese Schraube eignet sich laut DIN XXX für eine sichere Verbindung zwischen zwei unterschiedlichen Bauteilen.'

tokenize_and_lemma(string)
```

Output: ['schrauben',
 'eignen',
 'laut',
 'din',
 'sicher',
 'verbindung',
 'unterschiedlich',
 'bauteil']
 
 The preprocessing step is common to the Knowledge Seeker method and to the Scikit classifier methods, therefore, it can be seen as a mandatory NLP step.


## Knowledge Seeker

The Knowledge Seeker Method, originally described in [Knowledge Seeker - Ontology Modelling for Information Search and Management](https://link.springer.com/book/10.1007/978-3-642-17916-7), was applied in German Industry Norms ([DIN](https://en.wikipedia.org/wiki/Deutsches_Institut_f%C3%BCr_Normung)). This thesis debates the usage, the hurdles and algorithm behind this method. The results can be found in the knowledge_seeker.ipynb file.

We first load the classified dataframe, calculate the chi_squared matrix and the r_tc matrix. Based off the frequency values, chi-squared and r_tc are calculated as follows:

```python
observed = data_tokenized.copy()
col_total = total_fd.iloc[-1][0:]
row_total = total_fd['row_total']

N = total_fd['row_total'][-1:]
N = int(N)

expected = np.outer(row_total,col_total)/N
expected = pd.DataFrame(expected)
expected.columns = total_fd.columns[:]
expected.index = total_fd.index
expected = expected.drop(columns='row_total')
expected = expected.drop(expected.index[-1])

chi_squared_matrix = (((observed-expected)**2)/expected)

r_tc = observed/expected
```

We can build a dependency matrix as follows:

```python
dependency_matrix = r_tc.applymap(lambda x: 'negative' if x <= 1.0 else 'positive')
```

We can then generate ontology graphs for our labels for each of our domain areas:

Konstruktion:

![Konstruktion](https://user-images.githubusercontent.com/123315352/234392279-81a36bb9-ddc8-4500-8930-6e839c146a93.png)

Qualitaet:

![Qualitaet](https://user-images.githubusercontent.com/123315352/234392407-8f2a7308-1a0c-445a-b3ec-9eec0f2f541f.png)

Management:

![Management](https://user-images.githubusercontent.com/123315352/234392478-4f05b7b3-681d-4a6f-a964-7a0a75bd1b79.png)

Kostenrechnung:

![Kostenrechnung](https://user-images.githubusercontent.com/123315352/234392471-46833e22-85bf-47d3-82b7-334ddf08192b.png)

Logistik:

![Logistik](https://user-images.githubusercontent.com/123315352/234392475-89c9990b-4879-4e26-ba04-4e013072c0e4.png)

Forschung und Entwicklung:

![Forschung_und_Entwicklung](https://user-images.githubusercontent.com/123315352/234392469-e4ea517b-e024-4706-a551-963f66a81aa6.png)

Sicherheit:

![Sicherheit](https://user-images.githubusercontent.com/123315352/234392479-a345a90d-30de-4c99-8cbc-1d093642ff7c.png)

Fertigung:

![Fertigung](https://user-images.githubusercontent.com/123315352/234392483-11a0e3a5-2f08-4427-a6b7-79ceb887f13d.png)


Our final step is to classify DIN industrial standards in XML form. We can apply the following algorithm:

```python

def xml_extractor(xml_doc):

    dir = '/content/drive/MyDrive/ColabNotebooks/Norm_Classifier/XML_Norms/'
    file_dir = dir + xml_doc
    tree = ET.parse(file_dir)
    root = tree.getroot()
    text = []
    document_name = []
    tags = []
    for elem in tree.iter():
      att_dict = elem.attrib    
      if 'id' in att_dict.keys():
        for attrib, domain_area in sent_dict.items():        
          if attrib == elem.attrib['id']:         
            elem.attrib['subject'] = domain_area                  
                         
    tree.write('/content/drive/MyDrive/ColabNotebooks/Knowledge_seeker/Files/Classified_XML_sec/'+xml_doc, xml_declaration=True, method='xml', encoding="utf-16") 
    return 'Code ran sucessfuly!'

xml_extractor('30041688_DIN_EN_1993-3-1.xml')

``` 




## Scikit

The Scikit methods for text classification are broadly debated [here](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). We applied and disscused the diferences between these methods and the Knowledge Seeker method. The results can be found in the norm_classifier.ipynb file.

We first vectorize our data from the preprocessing step, using the Tf-IDF vectorizer:

```python
doc_shuffled = shuffle(docs)

X = doc_shuffled['TEXT']
y = doc_shuffled['LABEL']

vectorizer = TfidfVectorizer(analyzer = 'word', tokenizer=tokenize_and_lemma, ngram_range=(1,3),use_idf=True,min_df=0.15,sublinear_tf=True)
X_vec = vectorizer.fit_transform(X)
```


After vectorizing our text, we now have to decide which Scikit classifier we will use.

We will use the method described in the picture below.

![Top Down](https://user-images.githubusercontent.com/123315352/234392482-080bd5c3-9f56-4969-9864-aa97e4b5463d.png)

Classifier and parameter dictionary.
```python
model_params = {
    'LinearSVC': {
        'model': LinearSVC(dual=False, tol=1e-3),
        'params' : {
            'penalty':['l1','l2'],
            'C': [20,50,200]
        }  
    },
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [5,10,20,50,200],
            'kernel': ['rbf','linear']
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10,20,50,200]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1,0.5,1,50,100]
        }
    },
    'complement_multinomial': {
        'model': ComplementNB(),
        'params': {
            'alpha': [0.1,0.5,1,50,100]
        }
    },    
    'BernoulliNB': {
        'model': BernoulliNB(),
        'params': {
            'alpha': [0.1,0.5,1,50,100]
        }
    },
    'Perceptron': {
        'model': Perceptron(max_iter=100),
        'params': {
            'penalty':['l1','l2'],
            'alpha': [0.1,0.5,1,50,100]
        }
    },
    'PassiveAggressiveClassifier': {
        'model': PassiveAggressiveClassifier(max_iter=30),
        'params': {
            'C': [1,5,10,20,50,200],
            
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [5,10,20,25]            
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [5,10,20,25]
        }                   
    },
    'RidgeClassifier': {
        'model': RidgeClassifier(tol=1e-2, solver="auto"),
        'params' : {
            'alpha': [0.1,0.5,1,50,100]
        }
    } 
}
```


We then apply the RandomizedSearchCV method:

```python
labels = test_data_df['Label']
text = test_data_df['Sentence']

vect = vectorizer.transform(text)

scores = []
for model_name, mp in model_params.items():
  print(model_name)    
  clf =  RandomizedSearchCV(mp['model'], mp['params'], n_iter=4)  

  clf.fit(X_train.toarray(),y_train)

  
  y_pred =  clf.predict(vect.toarray())
  y_test = labels

  scores.append({
      'model': model_name,
      'best_score': clf.best_score_,
      'best_params': clf.best_params_,
      'precision': metrics.precision_score(y_test, y_pred,average = 'weighted', zero_division=0),
      'recall': metrics.recall_score(y_test, y_pred,average = 'weighted', zero_division=0),
      'f1': metrics.f1_score(y_test, y_pred, average = 'weighted', zero_division=0)
  }) 
      
df = pd.DataFrame(scores,columns=['model','best_score','precision','recall','f1','best_params'])
df['mean'] = df.mean(axis=1)
df.nlargest(4,['mean'])
```

Since this is a Randomized Search, the top results differ every time the algorithm runs. I will choose Complement Multinomial, KNeighborsClassifier, Naive Bayes Multinomial and SVM to be fine-tuned with GridSearchCV. The parameters will fluctuate around their order size in order to fine tune them.


```python
labels = test_data_df['Label']
text = test_data_df['Sentence']

vect = vectorizer.transform(text)

scores = []
for model_name, mp in model_params_ft.items():
  print(model_name)    
  clf =  GridSearchCV(mp['model'], mp['params'])
  clf.fit(X_train.toarray(),y_train)
  y_pred =  clf.predict(vect.toarray())
  y_test = labels

  scores.append({
      'model': model_name,
      'best_score': clf.best_score_,
      'best_params': clf.best_params_,
      'precision': metrics.precision_score(y_test, y_pred,average = 'weighted', zero_division=0),
      'recall': metrics.recall_score(y_test, y_pred,average = 'weighted', zero_division=0),
      'f1': metrics.f1_score(y_test, y_pred, average = 'weighted', zero_division=0)
  }) 
      
df_ft = pd.DataFrame(scores,columns=['model','best_score','precision','recall','f1','best_params'])
df_ft['mean'] = df_ft.mean(axis=1)
```


Finally, the classification occurs according to the following syntax, using ComplementNB with an alpha value of 0.5:

```python
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=1/4, random_state=42,stratify = y, shuffle = True)

classifier = ComplementNB(alpha=0.5).fit(X_train.toarray(),y_train)

```

Another interesting analysis is to use a learning curve to figure out how big our dataset must be:

```python
cv = ShuffleSplit(test_size=1/4, random_state=42)
scoring_function = 'F1 Value'

train_sizes, train_scores, test_scores = learning_curve(ComplementNB(alpha=0.5),X_vec,y, scoring= scorer,cv = cv,verbose = 1)
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(train_sizes, train_mean, label='Training Score')
plt.plot(train_sizes, test_mean, label='Cross-Validation score')

plt.fill_between(train_sizes,train_mean-train_std,train_mean+train_std, color ='#DDDDDD')
plt.fill_between(train_sizes,test_mean-test_std,test_mean+test_std, color = '#DDDDDD')

plt.title('Learning Curve').set_color('white') 
plt.xlabel('Training Size').set_color('white') 

plt.ylabel(str(scoring_function).title()+' Score').set_color('white') 
plt.legend(loc='best')

plt.tick_params(axis='x', colors='white')    
plt.tick_params(axis='y', colors='white')

```

![LearningCurve](https://user-images.githubusercontent.com/123315352/234392473-4c9518b0-3f67-41bf-86bf-7d5ff3c5efc7.png)