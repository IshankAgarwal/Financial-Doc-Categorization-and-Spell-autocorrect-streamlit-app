#%%writefile app.py
import streamlit as st
from io import StringIO
import PyPDF2
import nltk
from fuzzywuzzy import process

import numpy as np
import pandas as pd
import pickle
import spacy
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from xgboost import XGBClassifier


import pytesseract
from PIL import Image
nltk.download('punkt')
nltk.download('punkt_tab')

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#C:\Program Files\Tesseract-OCR\tesseract.exe

#--------------------------------------------------------------------------------------------------
# CAtegorisation part
#-----------------------------------------------------------------------------------------------------

def text_processing(text):
    # spaCy Engine
    nlp = spacy.load('en_core_web_lg')
    # Process the Text with spaCy
    doc = nlp(' '.join(text))

    # Tokenization, Lemmatization, and Remove Stopwords, punctuation, digits
    token_list = [
                  token.lemma_.lower().strip()
                  for token in doc
                  if token.text.lower() not in nlp.Defaults.stop_words and token.text.isalpha()
                 ]

    if len(token_list) > 0:
        return ' '.join(token_list)
    else:
        return 'empty'
     
def sentence_embeddings(sentence):
    words = word_tokenize(sentence)                                     # split the sentence into separate words
    model = Word2Vec.load("word2vec_model.bin")                         # load the trained model

    vectors = [model.wv[word] for word in words if word in model.wv]    # get the vectors of each words

    if vectors:
        return np.mean(vectors, axis=0)                                 # return the average of vectors

    else:
        return np.zeros(model.vector_size)                              # we set the model parameter in training ---> vector_size = 300
     
def prediction(sentence):
  preprocessed_text = text_processing(sentence)

  # Text Convert into Embeddings
  test_features = sentence_embeddings(preprocessed_text)

  test_features=test_features.reshape(1,-1)

  # load
  file_name = "xgb_model_fianncialdoc_categorisation.pkl"
  xgb_model_loaded = pickle.load(open(file_name, "rb"))

  test_pred=xgb_model_loaded.predict(test_features)

  target = {0:'Balance Sheets', 1:'Cash Flow', 2:'Income Statement', 3:'Notes', 4:'Others'}
  predicted_class = target[test_pred[0]]
  return predicted_class


#--------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

df= pd.read_csv("cleaned_financial_words.csv")
cfw_data=df
# convert to lower case
cfw_lower=[str(w).lower() for w in list(cfw_data["cleaned_financial_words"])]

def correct_financial_terms(text, term_list):
    # Tokenize and clean the text
    tokens = nltk.word_tokenize(str(text).lower())

    st.write(tokens)
    #print(tokens)
    corrected_text = []
    position=0
    changed_index=[]
    for token in tokens:
        if token in term_list:
            corrected_text.append(token)
            position+=1
        else:
            # Use fuzzy matching to suggest closest financial term
            match, score = process.extractOne(token, term_list)
            #print(match, score)
            if score > 90:  # Arbitrary threshold for matching
                corrected_text.append(match)
                position+=1
                changed_index.append(position)
            else:
                corrected_text.append(token)
                position+=1

    # Re-join tokens to form the corrected sentence
    return (changed_index,' '.join(corrected_text))











st.write('''# Financial documents categorization and Spellings Autocorrection App''')

file = st.file_uploader("Pick a file")
if file:
  #print(file.name)

  st.write(file.name)

  if file.name[-3:] =="txt":
    #bytes_data = uploaded_file.getvalue()
    #for line in file:
    #  st.write(line)
    stringio = StringIO(file.getvalue().decode("utf-8"))
    st.write("## Incorrect text file data")
    st.write(stringio)
    
    #-----------------------------------------------------------------------------------
    st.write("# Document category : ")
    st.write(prediction(stringio.read().encode()))
    #-------------------------------------------------------------------------------------

    highlight_indexes,corrected_text = correct_financial_terms(stringio.read().encode(), cfw_lower)

    words = corrected_text.split()

    # Apply highlighting to the specified words
    for index in highlight_indexes:
      if 0 <= index < len(words):  # Check if index is valid
        words[index-1] = f"<span style='background-color: yellow; font-weight: bold;'>{words[index-1]}</span>"

    # Reconstruct the paragraph with highlighted words
    highlighted_content = " ".join(words)

    st.write("## Corrected data")

    # Display the content in Streamlit with HTML rendering
    st.markdown(highlighted_content, unsafe_allow_html=True)




    #st.write(corrected_text)



  elif file.name[-3:]=="pdf":
    pdf_reader = PyPDF2.PdfReader(file)
    # Extract the content
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text() +"\n"
    # Display the content

    st.write("## Original file content")
    st.write(content)

    #-----------------------------------------------------------------------------------
    st.subheader("# Document category : ")

    styled_text = f"""
    <div style="font-size:24px; color:blue;">
        {prediction(content)}
    </div>
    """

    # Display the styled text
    st.markdown(styled_text, unsafe_allow_html=True)
    
    #-------------------------------------------------------------------------------------

    highlight_indexes,corrected_text = correct_financial_terms(content, cfw_lower)
    words = corrected_text.split()

    # Apply highlighting to the specified words
    for index in highlight_indexes:
      if 0 <= index < len(words):  # Check if index is valid
        words[index-1] = f"<span style='background-color: yellow; font-weight: bold;'>{words[index-1]}</span>"

    # Reconstruct the paragraph with highlighted words
    highlighted_content = " ".join(words)

    st.write("## Corrected Content")

    # Display the content in Streamlit with HTML rendering
    st.markdown(highlighted_content, unsafe_allow_html=True)
  
  elif (file.name[-3:]).lower() == "png" or (file.name[-3:]).lower() == "jpg" :

    st.title("Extract Text from Uploaded Image")

    # Upload an image
    #uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg"])
    uploaded_image=file
    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        
        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Extract text from the image using Tesseract OCR
        text = pytesseract.image_to_string(image)
        
        # Display the extracted text
        st.subheader("Extracted Text")
        st.write(text)
        #st.write("# Document category : ")

        st.subheader("# Document category : ")

        styled_text = f"""
        <div style="font-size:24px; color:blue;">
            {prediction(text)}
        </div>
        """

        # Display the styled text
        st.markdown(styled_text, unsafe_allow_html=True)
        #st.write(prediction(text))
        #-------------------------------------------------------------------------------------

        highlight_indexes,corrected_text = correct_financial_terms(text, cfw_lower)
        words = corrected_text.split()

        # Apply highlighting to the specified words
        for index in highlight_indexes:
          if 0 <= index < len(words):  # Check if index is valid
            words[index-1] = f"<span style='background-color: yellow; font-weight: bold;'>{words[index-1]}</span>"

        # Reconstruct the paragraph with highlighted words
        highlighted_content = " ".join(words)

        st.write("## Corrected Content")

        # Display the content in Streamlit with HTML rendering
        st.markdown(highlighted_content, unsafe_allow_html=True)
