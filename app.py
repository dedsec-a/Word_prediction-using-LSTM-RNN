import streamlit as st
import pickle 
import pandas as pd 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Loading the Model 
model = load_model('next_word_predection.h5')

# loading the Tokenezier 
#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


# Function for the Next Word
def predict_next_word(model , token , text , max_sequence_length):
    token_list = token.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length -1):]
    token_list = pad_sequences([token_list] , maxlen= max_sequence_length , padding= 'pre')
    predicted = model.predict(token_list , verbose= 0)
    predicted_word_index = np.argmax(predicted , axis= 1)
    for word , index in token.word_index.items():
        if index == predicted_word_index:
            return word
        else :
            return None



# Streamlit App
st.title("Next Word Predection with LSTM GRU")
input_text = st.text_input("Enter the Words" , "To be or not to be ")
if st.button("Predict next Word"):
    max_sequence_len = model.input_shape[1]+1 
    next_word = predict_next_word(model , tokenizer , input_text , max_sequence_len-1)
    st.write(f"Next word is {next_word}")



