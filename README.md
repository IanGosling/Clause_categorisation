# Clause Classification Project

## NON-TECHNICAL EXPLANATION
The purpose of this project was to experiment with different models and types of text encoding  for text classification in the legal context.<br>
This was trained and tested on a ten clause subset of a kaggle dataset The project considered three different models:
* [K-Nearest Neighbour](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) using [TFID Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) using [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
* [Simple Multi-Layer Perceptron (MLP)](http://neuralnetworksanddeeplearning.com/chap1.html) built in [Tensorflow Keras](https://www.tensorflow.org/guide/keras) using [TFID Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

In each case the model was trained using k-folds validation and I used [the Optuna Library](https://optuna.org/) for hyperparameter optimisation.
The goal of the model was to obtain the highest level of accuracy in classifying an unseen set of clauses to the correct classification.  
The most successful model was a simple two layer perceptron which achieved 88% accuracy, given the relatively small dataset this was a good result. With more time I would like to have experimented further with different combinations of text preprocessing steps, methods of vectorisation and models.
Primarily this project was a learning exercise and should be treated as such.
## DATA
There is a data sheet below.  In summary, the data was from a kaggle data set to be found [here](https://www.kaggle.com/datasets/bahushruth/legalclausedataset).
To limit processing time I filtered this dataset by picking the ten clause types with the highest number of examples.  This summary demonstrates two challenges in the data set:<br>
<ol>1. There are closely related clause types.  For example; indemnification-and-contribution,  indemnification, and indemnification-by-the-company. <br>
2. Some of the records have clause types with very few words in them - sometimes less than ten.  <br>
3. There is a relatively small number of total records, 2150, spread over a relatively large range of categories. </ol>
This relative scarcity of data and the close inter-relations of some of the categories makes this quite a challenging categorisation project.  

| Label                           | Record Count | Min Word Count | Max Word Count | Avg Word Count |
|---------------------------------|--------------|----------------|----------------|----------------|
| indemnification-and-contribution | 180          | 5              | 415            | 303            |
| indemnification                 | 210          | 4              | 418            | 247            |
| confidentiality                  | 220          | 6              | 416            | 242            |
| indemnification-by-the-company   | 230          | 4              | 430            | 218            |
| contribution                     | 180          | 50             | 424            | 327            |
| participations                   | 210          | 29             | 428            | 253            |
| arbitration                      | 240          | 2              | 408            | 212            |
| confidential-information         | 240          | 3              | 410            | 195            |
| capitalization                   | 200          | 13             | 422            | 275            |
| payment-of-expenses              | 240          | 5              | 412            | 200            |
Word counts are prior to text pre-processing.
The text of the clauses was preprocessed to:
* Remove the Clause title (Label) from the main body of the clause
* Removing Special Characters, low casing, removing punctuation, lemmatization and removing stop words.  This was done using the [Gensim Library](https://radimrehurek.com/gensim/parsing/preprocessing.html).
The data was then shuffled and split into Train and Test sets.  The same train (75%) and test sets (25%) were used by each model to avoid any unintentional bias.  
For the MLP labels were converted to integers. 

## MODEL 
I selected three Classifier models with increasing levels of complexity.  The goal was to understand what benefits were gained at increasing levels of complexity and computing resource.  In the case of Decision Trees when it became clear that the model was suffering from overfitting, I also ran a Random Forest Classifier which improved the results.
* Simple Model -  [K-Nearest Neighbour](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) using [TFID Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* More complex with different vectorisation - [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) using [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html)
* Simple neural network model - [Multi-Layer Perceptron](http://neuralnetworksanddeeplearning.com/chap1.html) built in [Tensorflow Keras](https://www.tensorflow.org/guide/keras) using [TFID Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).  In selecting this model I followed the advice of Google's training course of the subject found [here](https://developers.google.com/machine-learning/guides/text-classification/step-2-5).<ul/>
I considered different models of vectorisation having reviewed this paper: ['Systematic Comparison of Vectorisation Methods in Classification Context' Krzeszewska et al, Applied Sciences, May 22.](https://www.mdpi.com/2076-3417/12/10/5119)
<br>All models were trained with k-fold validation (k=5) to mitigate the relatively small data set. 

## HYPERPARAMETER OPTIMISATION
Description of which hyperparameters you have and how you chose to optimise them. 

## RESULTS
A summary of your results and what you can learn from your model 

You can include images of plots using the code below:
![Screenshot](image.png)

## CONTACT DETAILS
https://www.linkedin.com/in/ian-gosling/


# Model Card

See the [example Google model cards](https://modelcards.withgoogle.com/model-reports) for inspiration. 

## Model Description

**Input:** Describe the inputs of your model 

**Output:** Describe the output(s) of your model

**Model Architecture:** Describe the model architecture you’ve used

## Performance

Give a summary graph or metrics of how the model performs. Remember to include how you are measuring the performance and what data you analysed it on. 

## Limitations

Outline the limitations of your model.

## Trade-offs

Outline any trade-offs of your model, such as any circumstances where the model exhibits performance issues. 

# Datasheet

## Motivation
The original data set was created by a Data Science enthusiast for the benefit of other students of data science.  No commercial use is expected or intended.

## Composition
The data is Legal Clauses with labels crawled from an open source legal website.

| Label                           | Record Count | Min Word Count | Max Word Count | Avg Word Count |
|---------------------------------|--------------|----------------|----------------|----------------|
| indemnification-and-contribution | 180          | 5              | 415            | 303            |
| indemnification                 | 210          | 4              | 418            | 247            |
| confidentiality                  | 220          | 6              | 416            | 242            |
| indemnification-by-the-company   | 230          | 4              | 430            | 218            |
| contribution                     | 180          | 50             | 424            | 327            |
| participations                   | 210          | 29             | 428            | 253            |
| arbitration                      | 240          | 2              | 408            | 212            |
| confidential-information         | 240          | 3              | 410            | 195            |
| capitalization                   | 200          | 13             | 422            | 275            |
| payment-of-expenses              | 240          | 5              | 412            | 200            |
Word counts are prior to text pre-processing.
There is no personal or sensitive information in the data. 

## Collection process
The original dataset is distributed via Kaggle  [here](https://www.kaggle.com/datasets/bahushruth/legalclausedataset).  The top ten clauses by record count were used in this subset of the data.
The data was collected in 2021.

## Preprocessing/cleaning/labelling

The text of the clauses was preprocessed to:
* Remove the Clause title (Label) from the main body of the clause
* Removing Special Characters, low casing, removing punctuation, lemmatization and removing stop words.  This was done using the [Gensim Library](https://radimrehurek.com/gensim/parsing/preprocessing.html).
The data was then shuffled and split into Train and Test sets.  The same train (75%) and test sets (25%) were used by each model to avoid any unintentional bias.  
For the MLP labels were converted to integers. 
The raw csv files are included in this repositary.

## Uses
- Educational use only
- No commercial use 
- https://creativecommons.org/publicdomain/zero/1.0/

## Distribution
The subset of data used for these models is distribted in this Github repository.
 
## Maintenance
Data set is not maintained.
