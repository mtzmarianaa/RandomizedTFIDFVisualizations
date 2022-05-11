# This script takes the input data (many .cvs files) and outputs a matrix representation of the works


from bleach import clean
import pandas as pd # working with data frames
import numpy as np # algorithmic stuff
import sklearn # machine learning package
import nltk # nlp package
from sklearn.model_selection import GridSearchCV # to optimize parameters in the different models
import re # regex pacakge
from nltk.corpus import stopwords # stopwords so we can remove them later
import snowballstemmer # to stem the words
stemmer = snowballstemmer.stemmer('spanish');
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # to convert text into numeric vectors
import string # string variable, useful for object oriented programming
import time


# Useful
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        
def remove_shortComments(OriginalData, min_length):
    '''
    Given the initial data (comments + origin) it removes all those rows that correspond
    with comments that are "too short", defined by min_length
    '''
    print(len(OriginalData))
    for k in range(len(OriginalData)):
        if len(re.findall(r'\w+', OriginalData['Comments'].loc[k] )) < min_length:
            OriginalData.drop( index = k, inplace = True )
    OriginalData.reset_index()
    print(len(OriginalData))
    return OriginalData

        
def clean_data( text, language = 'spanish' ):
    '''
        This function cleanes and standarizes text, it removes punctuation,
        converts everything to lower case, stems the utterances and removes
        stop words, if given it can also replace certain words with others (optional)
        :param text: list
            text for which the standarization will take place
        :param replacement: DataFrame
            data frame with the replacement words (first column the word to remove, second
            column the word to substitute it with)
    '''
      
    te = list(text)
    new_text = []
    m = len(new_text)
    
    for k in range(m):
        t = new_text[k]
        t = t.replace( '\n',  '' )
        new_text[k] = t
    
    for c in te:
        c = str(c)
        new_text.append(''.join(w for w in c if w not in string.punctuation))
    
    # Remove punctuation
    new_text = [''.join(c for c in s if c not in string.punctuation) for s in new_text]
    
    # Remove numbers
    
    new_text = [ re.sub(r'[0-9]+', '', s) for s in new_text ]
    
    # Remove "characters I've found that are repeated and useless"
    new_text = [re.sub(r'[^\w\s]|(.)(?=\1)', '', s) for s in new_text]
    
    
    # Convert the titles to lowercase
    new_text = [ x.lower() for x in new_text ]
    
    # Remove stop words
    stop = stopwords.words(language)
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    new_text = [ x.replace(pat, '') for x in new_text ]
    new_text = [ x.replace(r'\s+', ' ') for x in new_text]

    for k in range(len(new_text)):
        sentence = new_text[k]
        stemmed_sentence = " ".join(stemmer.stemWords(sentence.split()))
        new_text[k] = stemmed_sentence
        
    return new_text 

# Read the multiple .cvs files with data from different sources

df_full_comments = pd.read_csv('data/comments_wOriginExcelsior.csv')
df_full_comments = df_full_comments.append(  pd.read_csv('data/comments_wOriginMaerker.csv'), ignore_index = True )
df_full_comments = df_full_comments.append(  pd.read_csv('data/comments_wOriginMilenio.csv'), ignore_index = True )
df_full_comments = df_full_comments.append(  pd.read_csv('data/comments_wOriginReforma.csv'), ignore_index = True )
df_full_comments = df_full_comments.append(  pd.read_csv('data/comments_wOriginUniversal.csv'), ignore_index = True )
df_full_comments.drop(df_full_comments.columns[[0]], axis=1, inplace = True)
print(df_full_comments.shape)
print( df_full_comments.head() )
df_full_comments = remove_shortComments(df_full_comments, 10)
# Clean
cleaned_text = clean_data( df_full_comments['Comments'], language = 'spanish' )
df_full_comments['Processed Comments'] = cleaned_text
# Save
df_full_comments.to_csv('data/FullAndClean.csv')

# Initiate the TD-IDF vectorizer objects
countvectorizer = CountVectorizer()
tfidfvectorizer = TfidfVectorizer()

# Convert the documents (text) to a matrix
count_wm = countvectorizer.fit_transform(df_full_comments['Processed Comments'])
tfidf_wm = tfidfvectorizer.fit_transform(df_full_comments['Processed Comments'])

count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()

df_countvect = pd.DataFrame(data = count_wm.toarray(), columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns = tfidf_tokens)


df_countvect['Origin'] = df_full_comments['Origin']
df_countvect['Origin'] = df_full_comments['Origin']
df_countvect.to_csv('data/Counts.csv')
df_tfidfvect.to_csv('data/TFIDF.csv')