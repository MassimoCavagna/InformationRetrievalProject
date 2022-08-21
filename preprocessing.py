import spacy
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from collections import defaultdict

def filter_doc(doc, nlp, pos_list = [
                               'PROPN',
                               'PRON',
                               'PUNCT',
                               'SCONJ',
                               'CCONJ',
                               'CONJ',
                               'DET',
                               'NUM',
                               'ADP',
                               'SPACE',
                               'SYM'
                               ]
              ):
  
  """
  This function uses spaCy to filter out the POS contained in the pos_list and takes the lemma.
  (see spaCy's doc for the complete list of POS and lemma)

  Parameters:
    - doc: the document in the form of pd.Series or dictionary containing the (textual) "content" and the "sentiment" expressed by it.
    - nlp: spacy.load("en_core_web_sm")
    - pos_list: the part of speech to be filtere out

  Return:
    a dictionary in the form:
    {
      sentiment : doc['sentiment']
      content: list of words not filtered out
      *sentiment_score: doc['sentiment_score']
    } 
  * if provided
  """
  
  content = ["#S"]
  for word in nlp(doc['content']):
    if word.pos_ not in pos_list and word.text[0] != '@':
      lemma = word.lemma_.lower()
      if len(lemma) > 1: 
        content.append(lemma)
  content.append("#E")
  to_return = {k:v for k,v in doc.items()}
  to_return['content'] = content
  return to_return


def skipgram(sequence, ws, postfix_size):
  """
  This function takes a sequence of words and computes the skipgrams with the given window size and postfix_size 
  Paramenters:
    sequence: the list containing the tokens that must be skipgrammed
    ws: the distance at which are looked words from the current one
    postfix_size: the number of words that will be taken in account after the window size space 
    
  Return:
    The list containing the list of n-grams with the skipped windows size, the tokens in each gram are separated by '~'
  """
  result = []
  for pos, token in enumerate(sequence, 1):
    for i in range(ws):
      start = min(pos+i, len(sequence))
      end = min(pos+i+postfix_size, len(sequence))
      if (end - start >= postfix_size):
        sk_gram = ( *[token], *sequence[start : end] )
        result.append("~".join(sk_gram))
  return result

def compute_skip_grams(data, ws = 1, postfix_size = 1):
  """
    This function compute the skipgram for each tweet in the dictionary 'data'. 
    
    Parameters:
      data: the dictionary containing the docs that must be skipgrammed.
      Each doc must be in the form:
      {
        sentiment : doc['sentiment']
        content: list of words
        *sentiment_score: doc['sentiment_score']
      } 
      * if provided
      ws: the distance at which are looked words from the current one
      postfix_size: the number of words that will be taken in account after the window size space 

    Return:
      The skipgrammed dictionary
  """
  sngram_te = {}

  for id, tweet in data.items():

    sk_list = skipgram(tweet['content'], ws, postfix_size)
    
    sngram_te[id] = {k: (tweet[k] if k != 'content' else sk_list) for k in tweet}
                             
  return sngram_te

def compute_tokens_frequency(data, freq: int = None):
  """
  This function first lists for each sentiment the tokens in the docs labeled with the same sentiment,
  Then computes for each token how many times it appears in the docs with the same sentiment.

  Paramenters:
    data: the dictionary of docs 
      Each doc must be in the form:
        {
          sentiment : doc['sentiment']
          content: list of words
          *sentiment_score: doc['sentiment_score']
        } 
    freq: frequency threshold under which tokens are discarded

  Return:
    a dictionary in the form
    {
      sentiment1:
                {
                  token1: {freq: freq1, sentiment_score: *score1}
                  token2: {freq: freq2, sentiment_score: *score2}
                  ...
                }
    
    }

  * if provided
  """
  sentiment_token = defaultdict(lambda: [])

  for k, doc in data.items():
    sentiment_token[doc['sentiment']] += [doc]
  sentiment_token = dict(sentiment_token)

  sentiment_score_flag = False
  if 'sentiment_score' in list(data.values())[0].keys():
    sentiment_score_flag = True
    l = lambda: {'freq':0, 'sentiment_score':0}
  else:
    l = lambda: {'freq': 0}

  sentiment_token_count = defaultdict(lambda: defaultdict(l))
  for s, docs in sentiment_token.items():
    for doc in docs:
      for token in doc['content']:
        sentiment_token_count[s][token]['freq'] += 1
        if sentiment_score_flag:
          sentiment_token_count[s][token]['sentiment_score'] = doc['sentiment_score']

  sentiment_token_count = {k : dict(d) for k, d in dict(sentiment_token_count).items()}

  if freq:

    def filter_tokens_by_frequency(data, freq):
  
    # This function filters out the tokes with a lower or equal freqency than freq
    
      for sentiment, docs in data.items():
        l = list(docs.items())
        for doc, value in l:
          if value['freq'] <= freq:
            docs.pop(doc)
    
    filter_tokens_by_frequency(sentiment_token_count, freq)

  return sentiment_token_count

def compute_inverse_document_frequency(tf):
  """
  This function computes the inverse document frequency from the frequency dictionary computed by compute_tokens_frequency
  Parameters:
    tf: the outcome of the compute_tokens_frequency function
  
  Returns:
    the idf value for each token
  """
  idf_tmp = defaultdict(lambda: defaultdict(lambda: 0))
  
  for sentiment, docs in tf.items():
    for token in docs:
      idf_tmp[token][sentiment] = 1

  N = len(tf.keys())
  
  idf = defaultdict(lambda: 1)

  for token in idf_tmp:
    idf[token] = N/(sum(idf_tmp[token].values()) + N)

  return idf

def compute_tf_idf(data, freq_to_filter: int = None):
  """
  This function combines the output of compute_tokens_frequency and compute_inverse_document_frequency to compute the tf.idf value for each token

  Parameters:
    data: the dictionary of docs 
      Each doc must be in the form:
        {
          sentiment : doc['sentiment']
          content: list of words
          *sentiment_score: doc['sentiment_score']
        }
    freq_to_filter: frequency threshold under which tokens are discarded
    * if provided
  Returns:
    a pd.DataFrame with tokens as columns and sentiments as index
    each entry is the tf.idf value
  """
  tf = compute_tokens_frequency(data, freq_to_filter)
  #filter_tokens_by_frequency(tf, freq_to_filter)
  idf = compute_inverse_document_frequency( tf = tf ) 

  sentiment_token_tfidf = pd.DataFrame(columns = idf.keys(), index = tf.keys(), dtype = float)

  for sentiment, tokens in tf.items():
    for token, d in tokens.items():
      sentiment_token_tfidf.loc[sentiment][token] = (d['freq'] * np.log( idf[token] + 1 )) * (d['sentiment_score'] if 'sentiment_score' in d.keys() else 1)
  
  return sentiment_token_tfidf.fillna(0)
