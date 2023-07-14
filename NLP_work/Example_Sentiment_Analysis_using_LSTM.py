#!/usr/bin/env python
# coding: utf-8

# # **Sentiment Analysis on the IMDB movie reviews**
# 
# ## **Context**
# 
# Movie reviews help users decide whether a movie is worth watching or not. A summary of the reviews for a movie can help a user make quick decisions within a small period of time, rather than spending much more time reading multiple reviews for a movie. Sentiment analysis helps in rating how positive or negative a movie review is. Therefore, the process of understanding if a review is positive or negative can be automated as the machine learns different techniques from the domain of Natural Language Processing.
# 
# ## **Objective**
# 
# The dataset contains 10,000 movie reviews. The objective is to do Sentiment Analysis(positive/negative) for the movie reviews using Deep Learning Sequential model Long short term Memory (LSTM) different techniques and observe the accurate results.
# 
# 
# ## **Data Dictionary**
# - **review:** reviews of the movies.
# - **sentiment:** indicates the sentiment of the review 0 or 1( 0 is for negative review and 1 for positive review)        

# ## **Importing the libraries**

# In[5]:


#!pip install wordcloud


# In[1]:


# Importing the required the libraries
import numpy as np
# To read and manipulate the data
import pandas as pd
pd.set_option('max_colwidth', None)

# To visualise the graphs
import matplotlib.pyplot as plt
import seaborn as sns

# Helps to display the images
from PIL import Image

import wordcloud
# Helps to visualize the wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ### **Loading the dataset**

# In[2]:


# Loading data into pandas dataframe
#reviews = pd.read_csv("/content/drive/MyDrive/cleaned_reviews.csv")
reviews = pd.read_csv("imdb_10K_sentimnets_reviews.csv")


# In[3]:


reviews.info()


# In[4]:


# Creating the copy of the data frame
data = reviews.copy()


# ## **Overview of the dataset**

# **View the first and last 2 rows of the dataset**

# In[5]:


data.head(2)


# In[6]:


data.tail(2)


# * Here, a sentiment value of **0 is negative**, and **1 represents a positive sentiment.**

# ### **Understand the shape of the dataset**

# In[7]:


# Print shape of data
data.shape               


# ### **Check the data types of the columns for the dataset**

# In[8]:


data.info()


# **Observations:**
# 
# * Data has 10000 rows and 2 columns.
# * Both the columns are object type.
# * There are no null values present in the dataset.

# ### **Checking for duplicate values**

# In[9]:


# checking for duplicate values
data.duplicated().sum()


# * There are no duplicate values present in the data since we are using the cleaned data from week 1.

# ## **Exploratory Data Analysis**

# **Word Cloud for cleaned Negative Reviews**

# In[10]:


# Creating word cloud for negative reviews

# Extracting the negative reviews i.e, sentiment = 0
negative_reviews = data[data['sentiment'] == 0]

# joining the negative reviews using space seperator, helps to convert the all rows into one string
words = ' '.join(negative_reviews['review']) 

# helps to remove the \n characters from the previous output
cleaned_word = " ".join([word for word in words.split()]) 


# In[11]:


# creating the wordcloud using the WordCloud() method
wordcloud = WordCloud(stopwords = STOPWORDS,
                      colormap = 'RdBu',
                      background_color = 'white',
                      width = 3000,
                      height = 2500
                     ).generate(cleaned_word) # The generate() function takes one argument of the text we created, helps to generate the wordcloud


# In[13]:


plt.figure(1, figsize = (8, 8))

# Using the .imshow() method of matplotlib.pyplot to display the Word Cloud as an image.
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# * The **even, bad, never, little, least, maybe, instead, waste, still, boring** were some of the important recurring words observed in the negative reviews.

# **Word Cloud for cleaned Positive Reviews**

# In[14]:


# Creating word cloud for positive reviews

positive_reviews = data[data['sentiment'] == 1]
# joining the negative reviews using space seperator, helps to convert the all rows into one string
words = ' '.join(positive_reviews['review'])
# helps to remove the \n characters from the previous output
cleaned_word = " ".join([word for word in words.split()])


# In[15]:


# creating the wordcloud using the WordCloud() method
wordcloud = WordCloud(stopwords = STOPWORDS,
                      colormap = 'RdBu',
                      background_color = 'white',
                      width = 3000,
                      height = 2500
                     ).generate(cleaned_word) # The generate() function takes one argument of the text we created, helps to generate the wordcloud


# In[16]:


plt.figure(1, figsize = (8, 8))

# Using the .imshow() method of matplotlib.pyplot to display the Word Cloud as an image.
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


#  **Character,story, well, good, best, great, scene,enjoy, interesting, wonderful** were some of the important words observed in the positive reviews

# In[17]:


# check the count of each labels
data['sentiment'].value_counts()    


# We can observe that classes are balanced.

# In[19]:


get_ipython().system('pip install keras')


# In[22]:


#pip install tensorflow


# In[23]:


# Helped to create train and test data
from sklearn.model_selection import train_test_split

# Metrics to evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Used to create the Sequential model
from keras.models import Sequential
import tensorflow
# Used to create the tokens from the text data
from tensorflow.keras.preprocessing.text import Tokenizer

# Helps to pad the sequences into the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Layers that are used to implement the LSTM model
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


# ### **Tokenizing and converting the reviews into numerical vectors**

# In[24]:


# Creating the tokenizer with 700 vocab size
tokenizer = Tokenizer(num_words = 700, split = ' ') 

tokenizer.fit_on_texts(data['review'].values)

# converting text to sequences
X = tokenizer.texts_to_sequences(data['review'].values)

# Padding the sequences
X = pad_sequences(X)


# **Model Building**

# In[25]:


model = Sequential()

# model will take as input an integer matrix of size (batch, input_length), and the largest integer (i.e. word index) in the input
# should be no larger than vocabulary size. Now model.output_shape is (None, input_length, 256), where `None` is the batch dimension.
# input_length is X_data[1] = 700 here.
model.add(Embedding(700, 120, input_length = X.shape[1]))

model.add(SpatialDropout1D(0.2))# dropping some values to decrease chances of over-fitting

 # return_sequences = True means each LSTM cell in it is outputting its value.The output of the layer is a sequence of outputs.
model.add(LSTM(150, dropout = 0.2, recurrent_dropout = 0.2))

model.add(Dense(2, activation = 'softmax'))

# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[26]:


print(model.summary())


# A sequential model is constructed by adding various layers to it.
# 
# - The first layer is the Embedding layer which transforms one-hot encoded sparse vectors into Word Embedding vectors. As the model continues to train, the weights of the Embedding layer are adjusted so that words with similar meanings are located closer together in the vector space, or have similar Word Embedding Vectors. For example, "orange" would be located near "tangerine" and "queen" would be near "empress." The vocabulary size is specified.
# 
# - The subsequent layer is an LSTM layer with 150 neurons. The input for this layer is a list of sentences, where each word has been converted to its corresponding Embedding vector and padded to have the same length. The activation function used is ReLU, which is widely used, but other relevant activation functions can also be used.
# 
# - To prevent bias, a dropout layer is employed to regulate the network.
# 
# - The final layer is a Dense layer which serves as the output layer and has 2 cells to perform classification, representing the 2 different categories in this example.
# 
# - The model is then compiled using the Adam optimizer and categorical cross-entropy. The Adam optimizer is currently the best choice for handling sparse gradients and noisy problems, and categorical cross-entropy is typically used when the classes are mutually exclusive, meaning each sample belongs to exactly one class.

# **Splitting the Data**

# In[27]:


# creating the target feature
y = pd.get_dummies(data['sentiment']) # why use get_dummies when this feature is already a 0 or 1?

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# **Training the model**

# In[1]:


# specifying the batch size 
batch_size = 32

# fitting the model on the training data with 10 epochs
#his = model.fit(X_train, y_train, epochs = 10, batch_size = batch_size, verbose = 'auto')


# **Plotting the model**

# In[ ]:


# accessing the accuracy from the his variable
plt.plot(his.history['accuracy'])
# setting the title 
plt.title('model training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# In[ ]:


# accessing the loss from the his variable
plt.plot(his.history['loss'])
# setting the title
plt.title('model training loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()


# **Observations:**
# - We can observe from the above graphs that the accuracy is got improved from the 0.75 to 0.91 in 10 epochs.
# - The training loss got reduced from the 0.50 to 0.20.

# **Evaluating the model on the test data**

# In[ ]:


model.evaluate(X_test,y_test)


# In[ ]:


ypred = model.predict(X_test)


# In[ ]:


ypred


# - The models is giving around 80% accuracy on the test data

# In[ ]:


# saving the model
model.save("/content/drive/MyDrive/model.h5")


# ##**Conclusion**
# 
# - In terms of accuracy, LSTM outperforms models such as Vader and TextBlob. However, the result is nearly identical to that of TF-IDF.
# - And, there is usually a trade-off between accuracy and computation.LSTM is good at remembering previous text sequences, and combining it with pretrained Word Embeddings like Word2Vec, GLoVe, and others can produce good sentiment analysis results.But training the model takes much longer than normal Supervised learning Algorithms like Random Forest using TF-IDF.
# 
