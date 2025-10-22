import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sentence_transformers import CrossEncoder
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from tqdm.auto import tqdm

embedder=SentenceTransformer('all-MiniLM-L6-v2') #SBERT Embeddings Generation

nltk.download('stopwords')
nltk.download('punkt_tab')      
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng')


df=pd.read_csv('DataNeuron_Text_Similarity.csv') #Loading Dataset


lemmatizer=WordNetLemmatizer() #Lemmatizer Initialization

def clean_text(s): #Text Cleaning Function
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def lowercase(s): #Lowercase Function
  return s.lower()

def remove_stopwords(s): #Stopword Removal Function
  words=word_tokenize(s)
  filtered_words=[word for word in words if word.lower() not in stopwords.words('english')]
  return ' '.join(filtered_words)


def remove_punc(s): #Punctuation Removal Function
  clean_text="".join(char for char in s if char not in string.punctuation)
  return clean_text

def get_wordnet_pos(tag): #POS Tag Conversion Function
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

def lemmatize_docs(s): #Lemmatization Function
  tokens=word_tokenize(s)
  tagged_tokens=pos_tag(tokens)

  lemmatized_sentence=[]
  for word, tag in tagged_tokens:
    if word.lower()=='are' or word.lower() in ['is','am']:
        lemmatized_sentence.append(word)  
    else:
        lemmatized_sentence.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    
  return ' '.join(lemmatized_sentence)

txt=remove_punc(df['text1'][0])
txt=lowercase(txt)
txt=remove_stopwords(txt)
txt=clean_text(txt)
txt=lemmatize_docs(txt)

df['pre_processed_text1']=df['text1'].apply(remove_punc).apply(lowercase).apply(remove_stopwords).apply(clean_text).apply(lemmatize_docs) #Pre-processing Text1 Column
df['pre_processed_text2']=df['text2'].apply(remove_punc).apply(lowercase).apply(remove_stopwords).apply(clean_text).apply(lemmatize_docs) #Pre-processing Text2 Column

new_df=df[['pre_processed_text1','pre_processed_text2']] #Creating New DataFrame with Pre-processed Text Columns

model=CrossEncoder('cross-encoder/stsb-roberta-base') #Cross Encoder Model Initialization
new_df['cross_encoder_score']=(model.predict([(new_df['pre_processed_text1'][i],new_df['pre_processed_text2'][i]) for i in range(len(new_df))])) #Generating Cross Encoder Similarity Scores

embeddings1=embedder.encode(new_df['pre_processed_text1'].tolist()) #Generating SBERT Embeddings
embeddings2=embedder.encode(new_df['pre_processed_text2'].tolist()) #Generating SBERT Embeddings
sbert_scores=[1 - distance.cosine(embeddings1[i], embeddings2[i]) for i in range(len(embeddings1))] #Calculating SBERT Similarity Scores
new_df['sbert_similarity_score']=sbert_scores

def token_jaccard(a, b): #Jaccard Similarity Function
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union

def seq_match_ratio(a, b): #Sequence Matcher Ratio Function
    return SequenceMatcher(None, a, b).ratio()

lexical_scores=[]
pairs=[(a, b) for a, b in zip(df['pre_processed_text1'], df['pre_processed_text2'])] #Creating Text Pairs for Lexical Similarity Calculation
for a, b in tqdm(pairs, desc="lexical"):
        lex_j = token_jaccard(a, b)
        seq_r = seq_match_ratio(a, b)
        lexical_scores.append(0.5 * lex_j + 0.5 * seq_r)

new_df['Jaccard_Score']=lexical_scores

def sbert_score(n): #Function to Ensure Non-negative SBERT Scores
    return abs(n)

new_df['sbert_similarity_score']=new_df['sbert_similarity_score'].apply(sbert_score)

similarity_scores=[]
for i in range(len(new_df)):
        similarity_scores.append((new_df['cross_encoder_score'][i]+new_df['sbert_similarity_score'][i]+new_df['Jaccard_Score'][i])/3) #Calculating Final Similarity Scores by Averaging

new_df['similarity_scores']=similarity_scores #Adding Final Similarity Scores to DataFrame