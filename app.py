from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder
from scipy.spatial import distance
import string
import re
import nltk
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from tqdm.auto import tqdm
import os

nltk.download('stopwords')
nltk.download('punkt_tab')      
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
#from sentence_transformers import CrossEncoder
lemmatizer=WordNetLemmatizer()
#cross_encoder_model=CrossEncoder('cross-encoder/stsb-roberta-base')
#sbert_embedder=SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def lowercase(s):
  return s.lower()

def remove_stopwords(s):
  words=word_tokenize(s)
  filtered_words=[word for word in words if word.lower() not in stopwords.words('english')]
  return ' '.join(filtered_words)


def remove_punc(s):
  clean_text="".join(char for char in s if char not in string.punctuation)
  return clean_text

def get_wordnet_pos(tag):
    if tag.startswith('J'):  
        return 'a'
    elif tag.startswith('V'):  
        return 'v'
    elif tag.startswith('N'):  
        return 'n'
    elif tag.startswith('R'):  
        return 'r'
    else:
        return 'n' 

def lemmatize_docs(s):
  tokens=word_tokenize(s)
  tagged_tokens=pos_tag(tokens)

  lemmatized_sentence=[]
  for word, tag in tagged_tokens:
    if word.lower()=='are' or word.lower() in ['is','am']:
        lemmatized_sentence.append(word)  
    else:
        lemmatized_sentence.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    
  return ' '.join(lemmatized_sentence)

def token_jaccard(a, b):
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union

def seq_match_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def similarity_score_generator(text1, text2):
    # Preprocess the texts
    text1 = remove_punc(text1)
    text1 = lowercase(text1)
    text1 = remove_stopwords(text1)
    text1 = clean_text(text1)
    text1 = lemmatize_docs(text1)

    text2 = remove_punc(text2)
    text2 = lowercase(text2)
    text2 = remove_stopwords(text2)
    text2 = clean_text(text2)
    text2 = lemmatize_docs(text2)

    # Cross-Encoder Score
    #cross_encoder_score=cross_encoder_model.predict([(text1, text2)])
    #print(cross_encoder_score)

    # SBERT Similarity Score
    #embedding1=sbert_embedder.encode([text1])[0]
    #embedding2=sbert_embedder.encode([text2])[0]
    #sbert_similarity_score=abs(1-distance.cosine(embedding1, embedding2))
    #print(sbert_similarity_score)

    # Jaccard Score
    lex_j=token_jaccard(text1, text2)
    seq_r=seq_match_ratio(text1, text2)
    jaccard_score=0.5*lex_j+0.5*seq_r
    print(jaccard_score)

    # Final Similarity Score
    final_similarity_score=jaccard_score
    print(final_similarity_score)

    return final_similarity_score

app=Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    data=request.json
    text1=data.get('text1', '')
    text2=data.get('text2', '')

    score=similarity_score_generator(text1, text2)

    return jsonify({'similarity score': float(score)})
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) 