Sentiment Analysis
==================

Train a sentiment classifier (Positive, Negative, Neutral) on a Roman Urdu dataset provided in Kaggle. 


Plan of work
============

-   Loading and cleaning the data.

-   Exploratory Data Analysis

-   Feature Engineering

-   Modeling and Evaluation

-   Challenges or Limitations

-   Future Work & Improvements

Loading and cleaning data
-------------------------

The data used for this use case is Roman Urdu data written in English
characters. The data contains 20229 entries of Positive, Negative and
Neutral sentiment text labeled accordingly.

  Positive   6013 entries
  ---------- --------------
  Negative   5286 entries
  Neutral    8928 entries


Based on the data exploration it is observed that the data has special
symbols and emoji's, etc. So basic text cleaning process followed to
clean the data before consuming in to the model. Few basic cleaning
applied for this use case are;

-   Tokenization

-   Remove Punctuation

-   Remove Alpha- numeric words

-   Convert words to lower case

-   Eliminating the words having character length lesser than two

-   Eliminating the rows wrongly labeled

Few other text pre-processing options;

-   Removing Tags -- There seems to be no tags available in the provided
    data.

-   Removal of Accented characters -- Not applied

-   Expanding Contractions -- This method do not apply for the current
    data, as the data is Urdu written in English characters.

-   Stemming and lemmatization -- This is not applicable for this data.

-   Removing Stop-words -- Observed few list of stop words for this data
    in Kaggle but applying it makes the data still sparser.

Exploratory Data Analysis 
-------------------------

To understand the distribution of the characters and words across the
reviews in the data created two features for each review.

-   Character Length

-   Word Count

The distribution of the character length and the word count across the
classes are studied. The data is highly skewed on the right with
minority of text with larger character length and word counts.


**Total No of Words: 268722**

**Size of the Vocabulary: 45124**

Feature Engineering 
-------------------

To build features on the review text few options considered for this
data.

-   Count Vectorizer

-   TF -- IDF Model

-   Embedding Matrix

For our data both Count Vectoriser and TF -- IDF performed similar and
no greater difference observed.
      
 Count Vectorizer 
===================
 **Max Features   **Training Accuracy**   **Validation Accuracy**   
  2500           0.6741                  0.6348                    
  1500           0.6264                  0.6154                    
  1000           0.5915                  0.5993                    

 TF - IDF 
===================
**Max Features   **Training Accuracy**   **Validation Accuracy**
  2500           0.6450                  0.6324                    
  1500           0.6490                  0.6111                    
  1000           0.5623                  0.5585                    

For Embedding Matrix, Keras Tokenizer text processing function is used
to convert the text to sequences. After converting the text to
sequences, padding applied to maintain the same input length. If the
text length is greater than the maximum sequence, the words after the
maximum sequence length are ignored for the features. This feature is
applied to LSTM network for text classification.

Modeling
--------

Few set of models created to address this text classification problem.

-   Traditional Machine Learning Models

-   ANN Implementation using Keras

-   LSTM Implementation using Keras

### Traditional Machine Learning Models

Initially baseline models are build using Random Forest and Logistic
Regression. Features considered for this model building is by using
Count Vectorizer and TF -- IDF approach. The features created are too
sparse for the model to learn. Also two additional features character
length and the word count are used for this model.

  Metrics            Random Forest                                                                       Logistic Regression
  ------------------ ----------------------------------------------------------------------------------- -----------------------------------------------------------------------------------
  Accuracy           0.64                                                                                0.65
  Precision          0.64                                                                                0.64
  Recall             0.64                                                                                0.658
  Confusion Matrix   ![](media/image4.png){width="1.6180555555555556in" height="1.3260597112860892in"}   ![](media/image5.png){width="1.6388888888888888in" height="1.3478718285214348in"}

### ANN Implementation using Keras

Based on the baseline model outcomes the same features are applied on an
Artificial Neural Network with the below mentioned simple architecture.
Many trials have been conducted on the model architecture to improve the
accuracy of the model but it is observed the model tend to over fit
after few epochs and the loss doesn't converge after a time period. The
loss curve and accuracy curve with the performance metrics are shown
below. ![](media/image6.png){width="3.373552055993001in"
height="2.4722222222222223in"}

  **Training Accuracy: 0.6646 , Validation Accuracy: 0.6225**                         
  ----------------------------------------------------------------------------------- -----------------------------------------------------------------------------------
  ![](media/image7.png){width="2.8055555555555554in" height="1.9083475503062117in"}   ![](media/image8.png){width="2.8680555555555554in" height="1.9120374015748032in"}

### LSTM Implementation using Keras

The ANN model outcome lead to a thought process to solve the problem
using a sequential network. Also by using the Count Vectorizer and TF --
IDF the importance is given to the presence of the word in a review
rather not to the context of the words. So here model architecture is
changed to have embedding layer followed by a LSTM layer to understant
the sequential data. Below is the architecture implemented and the
metrics for the same is shown in the table below. Here the validation
loss doesn't converge after a period of time and tries stay around the
same point. Here we observe a possibility of local minima.

![](media/image9.png){width="3.7000863954505685in"
height="1.8333333333333333in"}

  **Training Accuracy: 0.7993 , Validation Accuracy: 0.6612**                         
  ----------------------------------------------------------------------------------- -----------------------------------------------------------------------------------
  ![](media/image10.png){width="2.736111111111111in" height="1.8728094925634295in"}   ![](media/image11.png){width="2.861111111111111in" height="1.9427307524059492in"}

Several methods have been tried out to address the overfitting problem
in the above model as mentioned below;

1.  Simpler Architecture

    a.  Remove Layers

    b.  Decrease \# of neurons

2.  Data Augmentation - Artificially increase \# of training samples --
    Not tried out here.

3.  Early Stopping -- The model does not learn after a time.

4.  Dropout -- Helped to achieve this level of accuracy on training and
    validation.

5.  Regularization (Add penalty to error function) - L1 and L2

All the above-mentioned models perform much better when only the two
classes "Positive" and "Negative" in to considerations.

For instance, below are the tests conducted just with two class
"Positive" and "Negative";

                        Accuracy   Precision   Recall
  --------------------- ---------- ----------- --------
  Random Forest         0.77       0.77        0.77
  Logistic Regression   0.83       0.77        0.809

Challenges and Limitations
--------------------------

-   The data is not in the native Urdu language and this would have been
    better if it is native. This leads to the possibility of trying out
    on the pre-trained word vectors. In addition, in that case we can
    trial out modern architectures like BERT.

-   Knowledge of stop-words in this context is limited and does not
    always work well as it creates the data sparser.

-   Even after trying out different techniques to address overfitting of
    the network, the model does not learn on the data provided over a
    period.

-   Prevalence of data imbalance and data skewness is possibly a reason
    for model not to learn effectively.

Future Work & Improvements
--------------------------

-   The data can be transliterated to actual Urdu language text and
    apply pre-trained word vectors, which could possibly improve the
    performance of the model. Usage of transliteration libraries like
    Polyglot could convert this data to native text.

-   Try with Ensemble Methods based on the statistical nature of data
    distributions. This may possibly a solution to address sparse nature
    of the features.

-   Advanced Data Cleansing methods can optimally improve the
    performance of the model.

-   Apply advanced models like BERT after converting the text to native
    forms.


**Naïve Bayes:** results when only positive and negative classes are considered.

Positive:

Precision: 0.71 \| Recall: 0.55 \| Accuracy: 0.68

Precision: 0.71 \| Recall: 0.63 \| Accuracy: 0.70

Negative:

Precision: 0.66 \| Recall: 0.73 \| Accuracy: 0.66

Precision: 0.73 \| Recall: 0.79 \| Accuracy: 0.73
