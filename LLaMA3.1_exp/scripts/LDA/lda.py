import gensim
import nltk
import string

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from spellchecker import SpellChecker
from gensim.corpora import Dictionary

def stem_tokens(stop_words, claim):
    '''
    stop_words: a list of stop words
    '''    
    # all processed docs 
    all_processed_docs = []
    
    # initialize PorterStemmer
    p_stemmer = PorterStemmer()
    # initialize Wordnet lemmatizer
    lemmatizer = WordNetLemmatizer()
    # initialize spell checker 
    spell = SpellChecker()

    for idx, doc in enumerate(claim):
        # lower each document
        raw = doc.lower()

        # tokenize the raw 
        tokens = nltk.word_tokenize(raw)
        # find those words that may be misspelled
        misspelled = spell.unknown(tokens)
        # remove stopwords, words with a length less than 2, and digits
        for word in stop_words:
            tokens = list(filter(lambda a: a != word and len(a)>2 and (not a.isdigit()) and (not a in misspelled), tokens))
        
        # remove punctuations
        raw = ' '.join(tokens)
        raw = raw.translate(str.maketrans('', '', string.punctuation))
        # re-tokenize the raw 
        tokens = nltk.word_tokenize(raw)
    
        # lemmatize tokens
        lemmatized_tokens = [lemmatizer.lemmatize(i) for i in tokens]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in lemmatized_tokens]
        
        # append tokens
        all_processed_docs.append(stemmed_tokens)
        
        # if (idx % 10000 == 0):
        #     print (idx)
        
    return all_processed_docs

def lda_data_preprocess(claim, tfidf_model_path):
    en_stop = get_stop_words('en')
    en_stop = list(set(en_stop))
    en_stop = [_stop_word.lower() for _stop_word in en_stop]

    split_claim = stem_tokens(en_stop, claim)

    dct = Dictionary(split_claim)
    dict_dct = dict(dct)
    valid_tokens = set([dict_dct[t] for t in dict_dct])

    list_text_clean = []
    for text in split_claim:
        text = [word for word in text if word in valid_tokens]
        if (text != []):
            list_text_clean.append(text)

    # turn our tokenized documents into a (id: term) dictionary
    id2word = Dictionary(list_text_clean)
    # convert tokenized documents into a (id, frequency) matrix
    corpus = [id2word.doc2bow(text) for text in list_text_clean]
    # convert tokenized documents into a (id, tf-idf score) matrix
    tfidf_model = gensim.models.TfidfModel.load(tfidf_model_path)
    corpus_tfidf = tfidf_model[corpus]
    
    return corpus_tfidf

def lda_get_topics(claim, tfidf_model_path, lda_model_path):
    # Load a potentially pretrained LDA model
    lda_model = gensim.models.LdaMulticore.load(lda_model_path)

    corpus_tfidf = lda_data_preprocess(claim, tfidf_model_path)
    topics = lda_model.get_document_topics(corpus_tfidf)
    return topics