# Bag of Words Meets Bags of Popcorn

This is the kaggele competition. In this we are performing sentiment analysis on the dataset ( https://www.kaggle.com/c/word2vec-nlp-tutorial ) having  movies reviews.

Our Aim is to predict the movies reviews by training the given train data set and applying the predictions on the test dataset and unlabeled train data.

We are using "jupyter notebook with pyhton 3.5.0". jupyter notebook provides a favorable envirenment with having python shell.
python has some predefined packages and libraries which is used for machine learning.

Here we have used packages like  "Sklearn" , "Numpy" , "Pandas" , "nltk" , "bs4" , "OS","re".

# packages and there functions.

*In case if packages / libraries are not already available . we will first insatll them by using "pip" command' in command promt.
*eg:- pip install sklearn 

*we using following libraries versions:

 python: 3.5.0 (v3.5.0:374f501f4567, Sep 13 2015, 02:27:37) [MSC v.1900 64 bit (AMD64)]
 
 pandas: 0.23.4
 
 numpy: 1.15.0
 
 bs4: 4.6.3
 
 sklearn: 1.1.0
 
 nltk: 3.3
 
 re: 2.2.1

# importing libraries
import pandas as pd                                 # reading the .CSV file I/O (e.g. pd.read_csv)

import numpy as np                                  # data manipulation

from bs4 import BeautifulSoup                       # for pulling data out of HTML and XML files.

from sklearn.feature_extraction.text import CountVectorizer 
                                                    #extract features in a format supported by ML algorithms(such as text)
                                                    
from sklearn.ensemble import RandomForestClassifier # for classification and prediction 

import nltk                                         # removes unnecessary words from dataset

from nltk.corpus import stopwords

import os

import re 

# Reading dataset
After importing these packages we proceed to our next step of reading the dataset / file .
We have read "Train" , "Test" , "Unlabeledtrain " datasets using "pd.read_csv" meathod.

train dataset is having - 25000 rows and 3 columns ("id" , "sentiment" , "review")

test dataset is having - 25000 rows and 2 columns ("id" ,"review")

unlabeledTrain dataset is having - 50000 rows and 2 columns ("id" ,"review")


# cleaning dataset
After completion of first step we moved to pre-processing of dataset.
In this we removed the unnecessary words from the "review" column of the dataset. For this we have used "Beautifulsoup" and "stopwords"
meathod in our cleaning section of code.
we perform this operation on every dataset.

# Making bag of words
In this step we are creating the bag of features. we took upto 5000 features out of 25000.

# Train the classifier
Here we are doing training of our model on test dataset .
RandomforestClassifier form package Sklearn.ensemble is used to train the model .
Random forest is a ensembel learning meathod we use for classification problems.

# Predicting reviews on testing and unlabeled data.
 After fiting the model on train data. we are ready to make predictions on test and unlabeledTrain datasets.
 So, here we will pass the test and unlabled data for the prdiction to the trained model.
 
# Data fields:
 id - Unique ID of each review
 
 sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
 
 review - Text of the review
 
# Saving file in output file 
The final predictions result will be stored in as a "CSV" extension file.
Output.to_csv("file_name") meathod is used for saving the results into file.

# output file names are:
  test_data_predict_model.csv
  
  unlabeledTrain_output.csv

# Zip file includes following four files:
  *code file   : 1) Bag_of_Words_model.ipynb
  
  *output file : 2) test_data_predict_model.csv
                 3) unlabeledTrain_output.csv
               
  * text file  : 4)README.md
   
   
   
