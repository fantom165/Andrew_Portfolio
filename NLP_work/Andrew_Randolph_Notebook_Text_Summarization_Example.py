#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# 
# ## **Context**
# ----------------------------------------------------------------------------------
# 
# Text Summarization is the task of extracting the gist of a lengthy document in the form of a summary. There's a lot of subjectivity around what represents a good summary versus what's a poor summary. The subjectivity largely revolves around whether the entire context contained in the larger document has been extracted and is conveyed by the summary of the same. But machines of course do not work on subjectivity. Therefore, we need to train models that play their part in understanding Natural Language, and only then will our models be able learn to produce outputs that to our corresponding inputs can encapsulate the entire message in shorter text.
# 
# Document Summarizers have found a lot of use, for example, in stock brokerage websites that can show us news in short regarding particular stocks. This will help traders who trade based on fundamental analysis. Summarizers can also help in summarizing threads of conversations in several messaging services, for example, in Google Workspace. This way, one can be updated about progress in their workspaces for example, without having to spend a lot of time in going through the threads.
# 
# --------------------------------------------------------------------------------------
# 
# ## **Problem Statement**
# 
# -------------------------------------------------------------------------------------
# The objective of this project is to build a Sequential NLP model which is trained to summarize long sentences to produce shorter texts that efficiently encapsulate meaning.
# 
# -------------------------------------------------------------------------------------
# 
# ## **Dataset**
# 
# ----------------------------------------------------------------------------------------------
# WikiHow is a large-scale dataset using the online WikiHow (http://www.wikihow.com/) knowledge base.
# 
# The features of this dataset are:
# 
# - **headline**: Bold lines as summary
# - **title**: A one-line header about the purpose of the text article
# - **text**: The text article itself

# ### **Loading the dataset**
# - Extract "wikihow-summarization.zip"
# - Read "wikihowAll.csv"

# In[ ]:


#!unzip "/content/drive/MyDrive/Datasets/wikihow-summarization.zip"
get_ipython().system('unzip "/content/drive/MyDrive/Colab Notebooks/wikihow-summarization.zip"')


# In[ ]:


import pandas as pd
df = pd.read_csv('wikihowAll.csv')


# In[ ]:


df.head()


# In[ ]:





# ### **Dropping null values**
# - To preprocess the text we need to drop null values first

# In[ ]:


df.dropna(how='any',axis=0, inplace=True)
df.drop_duplicates(subset=['text'],inplace=True)
df.shape


# ### **Preprocess Text**
# Preprocess values of the text & headline columns:
# 
# - Remove unwanted characters
# - Convert text to lowercase
# - Remove unwanted spaces
# - Remove stopwords
# - Replace empty strings with Null
# - Drop null values from the dataframe

# In[ ]:


import re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def preprocess_text(df, column_name=''):

  # Select only alphabets
  df[column_name] = df[column_name].apply(lambda x: re.sub('[^a-zA-Z0-9_ \n\.]', '', x))


  # Convert text to lowercase
  df[column_name] = df[column_name].apply(lambda x: x.lower())

  # Strip unwanted spaces
  df[column_name] = df[column_name].apply(lambda x: x.strip())

  # Remove stopwords
  df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))

  # Replace empty strings with Null
  df[column_name].replace('', np.nan, inplace = True)

  # Drop Null values
  df = df.dropna()

  return df


# Use the `preprocess_text` function on the text and headline column:

# In[ ]:


df = preprocess_text(df, column_name='headline')
df = preprocess_text(df, column_name='text')


# ### **Adding the START and END token at the beginning and end of the headline**

# In[ ]:


df['headline'] = df['headline'].apply(lambda x : 'sostok '+ x + ' eostok')


# ### **Printing some rows from the text and headline columns**

# ### **Get the length of each headline and text and add a column for that**

# In[ ]:


df['len_headline'] = df['headline'].apply(lambda x: len(x.split(" ")))
df['len_text'] = df['text'].apply(lambda x: len(x.split(" ")))
df.head()


# In[ ]:


df.shape


# In[ ]:


#df['len_headline'] = df['headline'].apply(lambda x: len(x.split(" ")))
#df['len_text'] = df['text'].apply(lambda x: len(x.split(" ")))
#df.head()


# ### **Check the distribution of data**
# - This will help us in deciding the maximum length

# In[ ]:


import matplotlib.pyplot as plt
text_word_count = []
headline_word_count = []

# populate the lists with sentence lengths
for i in df['text']:
  text_word_count.append(len(i.split()))

for i in df['headline']:
  headline_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text':text_word_count, 'headline':headline_word_count})
length_df.hist(bins = 50)
plt.show()


# ### **Let's check the percentage of headlines below 80 words**

# In[ ]:


cnt=0
for i in df['headline']:
    if(len(i.split())<=80):
        cnt=cnt+1
print(cnt/len(df['headline']))


# ### **Let's check the percentage of text below 600 words**

# In[ ]:


cnt=0
for i in df['text']:
    if(len(i.split())<=600):
        cnt=cnt+1
print(cnt/len(df['text']))


# ### **For reducing data, we'll take headings where length > 50 and text where length > 100**

# In[ ]:


df = df[(df.len_headline >50) & (df.len_text >100)]
df.shape


# ### **Train test split**
# 
# Split the dataset into train and test set with a 80% to 20% ratio.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.text, df.headline, test_size = 0.2, random_state=42)

#del df why delete the df?


# ### **Print one sample**

# In[ ]:


# text
X_train[0]


# In[ ]:


# headline
y_train[0]


# ### **Initialize parameter values**
# - Set values for max_features, maxlen
# - max_features: Number of words to take from tokenizer(most frequent words)
# - maxlen: Maximum length of each sentence

# In[ ]:


max_features = 10000
maxlen_headline = 50
maxlen_text = 100


# ### **Applying the `tensorflow.keras` Tokenizer and getting indices for words**
# - Initialize Tokenizer object with number of words as 10000
# - Fit different tokenizer objects on headline and text column
# - Convert the text to sequence
# 

# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer

feature_tokenizer = Tokenizer(num_words= 10000)
feature_tokenizer.fit_on_texts(X_train)
X_train = feature_tokenizer.texts_to_sequences(X_train)
X_test = feature_tokenizer.texts_to_sequences(X_test)

print("Number of Samples in X_train:", len(X_train))       
print(X_train[0])

label_tokenizer = Tokenizer(num_words=max_features)
label_tokenizer.fit_on_texts(y_train)
y_train = label_tokenizer.texts_to_sequences(y_train)
y_test = label_tokenizer.texts_to_sequences(y_test)

print("Number of Samples in y_train:", len(y_train))       
print(y_train[0])


# ### **Pad sequences**
# - Pad each example in the 'post' configuration according to the maximum lengths for text and headlines respectively.

# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, maxlen = maxlen_text, padding='post')
X_test = pad_sequences(X_test, maxlen = maxlen_text, padding='post')

y_train = pad_sequences(y_train, maxlen = maxlen_headline, padding='post')
y_test = pad_sequences(y_test, maxlen = maxlen_headline, padding='post')


# ### **Vocab mapping**
# 
# Note: There is no word at the 0th index

# In[ ]:


feature_tokenizer.word_index


# In[ ]:


label_tokenizer.word_index


# ### **Set number of words**
# - Since the above 0th index doesn't have a word, add 1 to the length of the vocabulary

# In[ ]:


num_words_text = len(feature_tokenizer.word_index) + 1
print(num_words_text)

num_words_headline = len(label_tokenizer.word_index) + 1
print(num_words_headline)


# ### **Delete rows that contain only START and END token**

# In[ ]:


ind=[]
for i in range(len(y_train)):
    cnt=0
    for j in y_train[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_train=np.delete(y_train,ind, axis=0)
X_train=np.delete(X_train,ind, axis=0)

ind=[]
for i in range(len(y_test)):
    cnt=0
    for j in y_test[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_test=np.delete(y_test,ind, axis=0)
X_test=np.delete(X_test,ind, axis=0)


# ### **Building our Model**
# - We'll use an Encoder-Decoder architecture here
# - We shall define an Input with shape (maxlen_text, )
# - Then we shall add an Embedding layer
# - Going forward, we shall add our encoder LSTM layers
# - Here we are using Teacher Forcing, so therefore, the decoder LSTMs shall externally receive Target Inputs
# - Then a decoder Dense layer shall produce final Output from the Time Distributed Outputs of decoder LSTMs
# 

# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, TimeDistributed
from tensorflow import keras

hidden_dim = 300
embedding_dim = 100

# Encoder
encoder_inputs = Input(shape=(maxlen_text, ))

# Embedding layer
enc_emb =  Embedding(num_words_text, embedding_dim, trainable=True)(encoder_inputs) 

# Encoder lstm 1
encoder_lstm1 = LSTM(hidden_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# Encoder lstm 2
encoder_lstm2 = LSTM(hidden_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# Encoder lstm 3
encoder_lstm3 = LSTM(hidden_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

# Embedding layer
dec_emb_layer = Embedding(num_words_headline, embedding_dim, trainable=True) 
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Dense layer
decoder_dense = TimeDistributed(Dense(num_words_headline, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs) 

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Print model summary
model.summary()


# ### **Compile the model**
# - use optimizer as 'rmsprop'
# - use loss as 'sparse_categorical_crossentropy'

# In[ ]:


model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')


# ### **Add Callbacks**

# In[ ]:



from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)


# ### **Fit the mode**l
# - Training will take some time

# In[30]:


history = model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:],
                    epochs=20, callbacks=[es], batch_size=96,
                    validation_data=([X_test, y_test[:,:-1]], y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))


# ### **Plot the results**

# In[31]:


from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")
pyplot.legend()
pyplot.show()


# ### **Dictionary to Convert Index to word**

# In[32]:


reverse_target_word_index = label_tokenizer.index_word
reverse_source_word_index = feature_tokenizer.index_word
target_word_index = label_tokenizer.word_index


# ### **Inference**
# - In Inference Stage, we shall not be using Teacher forcing. We shall re use the output of previous decoder timestep which becomes the input of the next timestep.

# In[33]:


# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(hidden_dim,))
decoder_state_input_c = Input(shape=(hidden_dim,))
decoder_hidden_state_input = Input(shape=(maxlen_text, hidden_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
(decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c]) #dec_emb2

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


# In[33]:





# Function to implement inference

# In[34]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :]) + 2
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (maxlen_headline-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


# ## **Features to text**
# - Add functions to get text back from encoded text and headline

# In[35]:


def seq2headline(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


# In[36]:


for i in range(0,10):
    print("Text:",seq2text(X_train[i]))
    print("Original headline:", seq2headline(y_train[i]))
    print("Predicted headline:", decode_sequence(X_train[i].reshape(1, maxlen_text)))
    print("\n")


# Nothing Here

# **OBSERVATIONS**:  
# * This model generates a very high number of trainable parameters (57MM plus), thus utilizing a lot of compute power.  
# * The original **dataset** (text and headlines) seems to be missing stopwords, thus the sentences do not read comprehensibly.  
# * **Increasing the size of the dataset** -We took only 10k points due to computational constraints. If we have high-end GPUs available this model can be trained on a much bigger dataset thereby improving the results.
# * Since we stopped the training after only a few epocs, the training loss is still decreasing. Therefore, the model's performance would increase if allowed to train further.  
# * **Use of Bidirectional LSTMs**- We discussed earlier that for Encoder decoder with the attention we make use of Bidirectional LSTMs. However we built our model with unidirectional LSTM.

# **RECOMMENDATIONS:**  
# * Train the model over many more epocs, but using a vocab including all stopwords.  
# * Train a model with more points, which would take more processing time.  
# * Perhaps train the model with a max summary words of 50 or less for ease of understanding long documents, but with the understanding that the fewer the max summary words, the worse the model will do at predicting correctly.
