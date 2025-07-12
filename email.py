import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

cv = pickle.load(open('cv.pkl','rb'))
model = pickle.load(open('mnb.pkl','rb'))

st.title('Spam Email and SMS Classifier')


ps=PorterStemmer()
def transform_text (text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
        y.append(i)
    text= y[:]
    y.clear()
    for i in text:
      y.append(ps.stem(i))
    return " ".join(y)


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://64.media.tumblr.com/3f43eb52e948ec149191d5558444e132/tumblr_o36og5L4VW1rnqolfo1_500.gif");
        background-size: cover;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


input_sms = st.text_input('Enter')
if st.button('Predict'):
    transformed_sms  = transform_text(input_sms)
    vector_input = cv.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result==1:
     st.header('Spam')
    else:
     st.header('Not Spam')