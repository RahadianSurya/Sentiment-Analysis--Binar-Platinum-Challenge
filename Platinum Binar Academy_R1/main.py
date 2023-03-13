import re
import sqlite3
import pandas as pd
import numpy as np
import joblib

#For Flask
from flask import Flask, jsonify
#Import library for tokeize, stemming, and stopwords
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords as stopwords_scratch

#Import library for sklearn model sentiment analysis
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
# Import library for Tensorflow Model Sentiment Analysis
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.models import load_model
# Import library for Flask
from flask import Flask, request, jsonify,render_template
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
# !pip install unidecode
from unidecode import unidecode
#Swagger UI Definition
app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title' : LazyString(lambda: 'Sental (Sentiment Analysis)'),
    'version' : LazyString(lambda : '1.0.0'),
    'description': LazyString(lambda : 'Data menganalisa suatu sentiment'),
},
    host = LazyString(lambda: request.host)
)

swagger_config = {
    'headers': [],
    'specs': [
        {
        'endpoint': 'docs',
        'route': '/docs.json',
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/docs/'
}
swagger = Swagger(app, template=swagger_template,
                 config = swagger_config)

#Connect db and csv
conn = sqlite3.connect('data/output.db', check_same_thread=False)
df_alay = pd.read_csv('data/new_kamusalay.csv', names=['alay', 'cleaned'], encoding= 'latin-1')
df_raw = pd.read_csv('data/train_preprocess.tsv', sep='\t', names=['Text', 'Sentiment'])
df_raw.drop_duplicates()

#Define and execute query for unexistence data tables
#Tables will contain fields with dirty text (text & file) and cleaned text (text & file)
conn.execute('''CREATE TABLE IF NOT EXISTS data_text_sk (text_id INTEGER PRIMARY KEY AUTOINCREMENT, Text varchar(255), Sentiment varchar(255));''')
conn.execute('''CREATE TABLE IF NOT EXISTS data_file_sk (text_id INTEGER PRIMARY KEY AUTOINCREMENT, Text varchar(255), Sentiment varchar(255));''')
conn.execute('''CREATE TABLE IF NOT EXISTS data_text_tf (text_id INTEGER PRIMARY KEY AUTOINCREMENT, Text varchar(255), Sentiment varchar(255));''')
conn.execute('''CREATE TABLE IF NOT EXISTS data_file_tf (text_id INTEGER PRIMARY KEY AUTOINCREMENT, Text varchar(255), Sentiment varchar(255));''')

list_stopwords = stopwords_scratch.words('indonesian')
list_stopwords_en = stopwords_scratch.words('english')
list_stopwords.extend(list_stopwords_en)
list_stopwords.extend(['ya', 'yg', 'ga', 'yuk', 'dah', 'baiknya', 'berkali', 'kali', 'kurangnya', 'mata', 'olah', 'sekurang', 'setidak', 'tama', 'tidaknya'])

#Add External Stopwords
f = open("stopwords/tala-stopwords-indonesia.txt", "r")
stopword_external = []
for line in f:
    stripped_line = line.strip()
    line_list = stripped_line.split()
    stopword_external.append(line_list[0])
f.close()
list_stopwords.extend(stopword_external)
stopwords = list_stopwords

#Creating function for Cleansing Process
def lowercase(text): # Change uppercase characters to lowercase
    return text.lower()
def special(text):
    text = re.sub(r'\W', ' ', str(text), flags=re.MULTILINE)
    return text
def single(text):
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text, flags=re.MULTILINE)
    return text
def singlestart(text):
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text, flags=re.MULTILINE)
    return text
def mulspace(text):
    text = re.sub(r'\s+', ' ', text, flags=re.MULTILINE)
    return text
def rt(text):#Removing RT
    text = re.sub(r'rt @\w+: ', ' ', text, flags=re.MULTILINE)
    return text
def prefixedb(text):#Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text, flags=re.MULTILINE)
    return text
def misc(text):
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))|([#@]\S+)|user|\n|\t', ' ', text, flags=re.MULTILINE)
    return text

#Mapping for kamusalay
alay_mapping = dict(zip(df_alay['alay'], df_alay['cleaned']))
def alay(text):
    wordlist = text.split()
    text_alay = [alay_mapping.get(x,x) for x in wordlist]
    clean_alay = ' '.join(text_alay)
    return clean_alay

def stopwrds(text): 
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    output_sw = ' '.join(tokens_without_sw)
    return output_sw

#Function for text cleansing
def cleansing(text):
    text = lowercase(text)
    text = special(text)
    text = single(text)
    text = singlestart(text)
    text = mulspace(text)
    text = rt(text)
    text = prefixedb(text)
    text = misc(text)
    text = alay(text)
    # text = stopwrds(text)
    return text

#Sklearn Neural Network Analysis Sentiment
#Load the sklearn Model

model_NN = joblib.load(r"C:\Users\rahad\OneDrive\Documents\Binar Machine Learning - Platinum\Platinum Binar Academy_R1\data\MLP_tfidf_ngram12_no_sw_882")


#Function for CSV Sklearn Analysis
def sentiment_csv_nn(input_file):
    column = input_file.iloc[:, 0]
    print(column)
    
    #Define and execute query for insert cleaned text and sentiment to sqlite database
    for data_file in column:
        data_clean = cleansing(data_file)
        sent = model_NN.predict(pd.DataFrame([data_clean])[0])[0]
        # sent = 'Positive' if sent == 2 else 'Neutral' if sent ==1 else 'Negative' if sent ==0 else 'None'
        query = "insert into data_file_sk ('Text', 'Sentiment') values (?, ?)"
        val = (data_clean,str(sent))
        conn.execute(query, val)
        conn.commit()
        print(data_file)

#Create Homepage
@swag_from(r"C:\Users\rahad\Downloads\Platinum Binar Academy\docs\welcome_pages.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def get():
    return "Welcome to Sental Dashboard"

#Text Analysis Sklearn
#Input text to analyze
@swag_from(r"C:\Users\rahad\Downloads\Platinum Binar Academy\docs\text_sklearn.yml", methods=['POST'])
@app.route('/text_sklearn', methods=['POST'])
def text_sentiment_sklearn():
    #Get text from user
    input_text = str(request.form['text'])
    
    #Cleaning text
    output_text = cleansing(input_text)
    
    #Model Prediction for Sentiment Analysis
    sent = model_NN.predict(pd.DataFrame([output_text])[0])[0]
    # sent = 'Positive' if sent == 2 else 'Neutral' if sent ==1 else 'Negative' if sent ==0 else 'None'
    
    # Define and execute query for insert cleaned text and sentiment to sqlite database
    query = "insert into data_text_sk (text,sentiment) values (?, ?)"
    val = (output_text,str(sent))
    conn.execute(query, val)
    conn.commit()
    
    #Define API Response
    json_response = {
        'description': "Analysis Sentiment Success!",
        'F1 on test set': "MLP Classifier with Tfidf",
        'text' : output_text,
        'sentiment' : str(sent)
    }
    response_data = jsonify(json_response)
    return response_data

#Endpoint for File Analysis SKLearn
@swag_from(r"C:\Users\rahad\Downloads\Platinum Binar Academy\docs\file_sklearn.yml", methods=['POST'])
@app.route('/file_sklearn', methods=['POST'])
def file_sentiment_sk():
    #Get File
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')
    
    #Cleaning file
    sentiment_csv_nn(datacsv)
    
    #Define API response
    select_data = conn.execute("SELECT * FROM data_file_sk")
    conn.commit
    data = [
        dict(text_id=row[0], text=row[1], sentiment=row[2])
    for row in select_data.fetchall()
    ]
    
    return jsonify(data)

#Tensorflow LSTM Model Analysis Sentimen
#Load the Tensorflow Model
model = load_model(r"C:\Users\rahad\OneDrive\Documents\Binar Machine Learning - Platinum\Platinum Binar Academy_R1\data\LSTM_89.h5")
tokenizer = joblib.load(r"C:\Users\rahad\Downloads\Platinum Binar Academy\data\tokenizer.pickle")

#Model Prediction
#Create Function for Sentiment Prediction
def predict_sentiment(text):
    sentiment_tf = ['negative', 'neutral', 'positive']
    text = cleansing(text)
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=96)
    prediction = model.predict(tw)
    polarity = np.argmax(prediction[0])
    return sentiment_tf[polarity]

def sentiment_csv_tf(input_file):
    column = input_file.iloc[:, 0]
    print(column)
    
    # Define and execute query for insert cleaned text and sentiment to sqlite database
    for data_file in column:
        data_clean = cleansing(data_file)
        sent = predict_sentiment(data_clean)
        query = "insert into data_file_tf ('Text', 'Sentiment') values (?, ?)"
        val = (data_clean,sent)
        conn.execute(query, val)
        conn.commit()
        print(data_file)

#Endpoint for Text Analysis Tensorflow
#Input text to analyze
@swag_from(r"C:\Users\rahad\Downloads\Platinum Binar Academy\docs\text_tensorflow.yml", methods=['POST'])
@app.route('/text_tensorflow', methods=['POST'])
def text_sentiment_tf():
    #Get text from user
    input_text = str(request.form['text'])
    
    #Cleansing text
    output_text = cleansing(input_text)
    
    #Model Prediction for Sentiment Analysis
    output_sent = predict_sentiment(output_text)
    
    #Define and execute query for insert cleaned text and sentiment to sqlite database
    query = "insert into data_text_tf (Text,Sentiment) values (?, ?)"
    val = (output_text,output_sent)
    conn.execute(query, val)
    conn.commit()
    
    #Define API response
    json_response = {
        'description': "Analysis Sentiment Success!",
        'text' : output_text,
        'sentiment' : output_sent
    }
    response_data = jsonify(json_response)
    return response_data

#Endpoint for File Analysis Tensorflow
@swag_from(r"C:\Users\rahad\Downloads\Platinum Binar Academy\docs\file_tensorflow.yml", methods=['POST'])
@app.route('/file_tensorflow', methods=['POST'])
def file_sentiment_tf():
    #Get file
    file = request.files['file']
    try:
            datacsv = pd.read_csv(file, encoding='iso-8859-1')
    except:
            datacsv = pd.read_csv(file, encoding='utf-8')
    
    #Cleaning file
    sentiment_csv_tf(datacsv)
    
    #Define API response
    select_data = conn.execute("SELECT * FROM data_file_tf")
    conn.commit
    data = [
        dict(text_id=row[0], text=row[1], sentiment=row[2])
    for row in select_data.fetchall()
    ]
    
    return jsonify(data)

    

if __name__ == '__main__':
    app.run()
