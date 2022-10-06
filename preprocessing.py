import spacy
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer



def filter_doc(doc, nlp, pos_list = [
                               'ADJ',
                               'ADV',
                               'AUX',
                               'INTJ',
                               'NOUN',
                               'PART',
                               'SCONJ',
                               'VERB',
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
  # Create list of word tokens after removing stopwords  
  #content = ["#S"]
  content = []
  for word in nlp(doc['content']):
    if word.pos_ in pos_list:
      lemma = word.lemma_.lower()
      if len(lemma) > 1 and lemma != "'s": 
        content.append(lemma)
  #content.append("#E")
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


def tfidf_vectorize(data, binary = False):
  """
  This function compute the tfidf vectorization of the corpus of documents passed by data and the related labels
  Params:
     data : the dictionary of docs 
      Each doc must be in the form:
        {
          sentiment : doc['sentiment']
          content: list of words
          *sentiment_score: doc['sentiment_score']
        }
    * if provided
    
  Return:
    - X : The matrix containing the tfidf of each document over the different features
    - y : the list of labels
  """

  corpus = [" ".join(doc["content"]) for doc in data.values()]
  y = [doc["sentiment"] for doc in data.values()]


  
  
  vectorizer = TfidfVectorizer()
  if binary:
    vectorizer = TfidfVectorizer(binary = True,)

  X = vectorizer.fit_transform(corpus)

  return X, y, vectorizer

def plot_history(histories : list, title : str, same_figure = False, group_names: list = None, figsize = (20,20)):
  """
  This function is used to easily plot the history returned by any model in the form of a dictionary.
  For each metric it plots a lineplot describing the model's trend through all the epochs
  Parameters:
  -histories: a list of histories or a dict containing different lists of histories, one for each key
  -title: the title of the figure
  -same_figure: if the result must be contained in a single figure, or it must just be plotted without
                intestation. Useful to plot multiple subplots in the same figure
  -group_names: a list containing the name of each data in the histories dictionary (must match the lenght of histories).
                By default it enumerates from 0 the different lines (one ofr each item in histories)
  -figsize: the size of the resulting figure
  Return: None
  """
  if same_figure:
    fig = plt.figure(figsize = figsize)
    fig.suptitle(title)

  
  df = pd.DataFrame()

  if (group_names != None) and (len(histories) != len(group_names)):
    raise Exception('The lenghts must be the same')

  for history, i in zip(histories, group_names if group_names != None else list(range(len(histories)))):
    if type(history) != dict:
      history = history.history

    keys, val_keys = [k for k in history.keys() if "val_" not in k], [k for k in history.keys() if "val_" in k]

    data = pd.DataFrame({k : history[k] for k in keys}, columns = keys)
    data["type"] = "T_" + str(i)
    data["epoch"] = list(range(len(data["type"])))

    val_data = pd.DataFrame({k.replace("val_", "") : history[k] for k in val_keys}, columns = keys)
    val_data["type"] = "V_" + str(i)
    val_data["epoch"] = list(range(len(val_data["type"])))

    if df.empty:
      df = pd.concat([data, val_data]).reset_index(drop=True)
    else:
      tmp = pd.concat([data, val_data]).reset_index(drop=True)
      df = pd.concat([df, tmp]).reset_index(drop=True)
    sns.set_style("darkgrid")
    
  df.sort_values(by=['type'], inplace = True)
  df.reset_index(drop=True)
  for i, k in enumerate(df.columns[0:-2]):
    n, is_val_empty = ((df.shape[0]/2)-1, False) if len(df[df.type.str.contains('V', case=False)]) > 0 else (df.shape[0]-1, True)
    plt.subplot((len(df.columns[0:-2])//3)+1, 3, 1 + i)
    plt.title(k)

    if group_names == None:
      sns.lineplot(data = df.iloc[:int(n)], x = "epoch", y = k, hue = "type", palette = sns.color_palette(["blue"]*len(histories), len(histories)), legend = i == (len(df.columns[0:-2])-1))
      if not is_val_empty:
        sns.lineplot(data = df.iloc[int(n+1):], x = "epoch", y = k, hue = "type", palette = sns.color_palette(["red"]*len(histories), len(histories)), legend = i == (len(df.columns[0:-2])-1))
    
    else:
      sns.lineplot(data = df.iloc[:int(n)], x = "epoch", y = k, hue = "type", palette = sns.color_palette("PuBu", len(histories)), legend = (i == (len(df.columns[0:-2])-1))) 
      if not is_val_empty:
        sns.lineplot(data = df.iloc[int(n+1):], x = "epoch", y = k, hue = "type", palette = sns.color_palette("OrRd", len(histories)), legend = (i == (len(df.columns[0:-2])-1)))
    if (i == (len(df.columns[0:-2])-1)):  
      plt.legend(bbox_to_anchor=(1.02, 0.73), loc='upper left', borderaxespad=0)  
