from flask import Flask, render_template, request
import pickle
import os, sys
from exception import CustomException
from logger import logging
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

### loading model and scalar object
logging.info("Reading model object from model.sav file")
model=pickle.load(open('Models\model.sav', 'rb'))
logging.info("Reading Vectorizer object from vectorizer.sav file")
vectorizer=pickle.load(open('Models\Tfidf.sav', 'rb'))

stemmer= PorterStemmer()

def text_transformer(in_text):
    ### lowering the text
    in_text=in_text.lower()
    ### seperating each word
    in_text=nltk.word_tokenize(in_text)
    ### removing special characters from in_text
    temp_list=[]
    for word in in_text:
        if word.isalnum():
            temp_list.append(word)
    in_text= temp_list.copy()
    temp_list.clear()
    
    ### removing stopwords and punctuation marks from in_text
    for word in in_text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            temp_list.append(word)
    in_text=temp_list.copy()
    temp_list.clear()
    
    ### Stemming the words to get base form of words
    for word in in_text:
        temp_list.append(stemmer.stem(word))
    ### joining all words in list and returning complete sentence/ document
    return " ".join(temp_list)


app=Flask(__name__)

### creating home route
logging.info("Creating Home route")
@app.route("/")
def home():
    return render_template("home.html")


# Creating Prediction route
logging.info("Creating Prediction route")
@app.route('/prediction',methods=['POST'])
def prediction():
    try:
        logging.info("Getting data from the web form for prediction")
        data=[str(x) for x in request.form.values()]
        transformed_text=text_transformer(data[0])
        vectorized_text=vectorizer.transform([transformed_text])
        output=model.predict(vectorized_text)
        if output==1:
            logging.info("Model Prediction is: Message is Spam.")
            output= "Message is Spam."
        else:
            logging.info("Model Prediction is: Message is ham.")
            output="Message is ham."
        logging.info("Returning model prediction to web application")
        return render_template("home.html",prediction_value="Model Prediction: {}".format(output))
    except Exception as e:
        raise CustomException(e, sys)

### Note check log file for the server link or paste this in browser: http://127.0.0.1:5000
if __name__=='__main__':
    app.run()