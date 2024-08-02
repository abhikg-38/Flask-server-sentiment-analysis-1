import pandas as pd;
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

  
def preprocess(text):
    words=word_tokenize(text.lower())
    words=[lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

app=Flask(__name__)
@app.route('/analyze', methods=['POST'])
def analyze():
    data=request.get_json()
    df=pd.DataFrame(data,columns=['review'])
    df['cleaned_review']=df['review'].apply(preprocess)
    tot_pos =0
    tot_neg=0
    tot_neu=0
    count=0
    for review in df['cleaned_review']:
        count+=1
        analysis=analyzer.polarity_scores(review)
        tot_pos+=analysis['pos']
        tot_neg+=analysis['neg']
        tot_neu+=analysis['neu']
    avg_pos=tot_pos/count
    avg_neg=tot_neg/count
    avg_neu=tot_neu/count
    averages={'average_positive': avg_pos,
            'average_negative': avg_neg,
            'average_neutral':avg_neu}
    return jsonify({'status':'success','averages':averages})
   

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5001)