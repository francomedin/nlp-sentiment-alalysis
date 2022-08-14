import re
import nltk
import spacy
import unicodedata

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
#stopword_list = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')


def remove_html_tags(text):
    text_without_tags = BeautifulSoup(text, "html.parser").text
    return text_without_tags


def stem_text(text):
    porter = PorterStemmer()
    text_stemmized = " ".join([porter.stem(word) for word in tokenizer.tokenize(text)]) 
    return text_stemmized


def lemmatize_text(text):
    text_lemmatized = nlp(text)
    text_lemmatized = " ".join([word.lemma_ for word in text_lemmatized])
    return text_lemmatized


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
# Did not work --> text = " ".join(word if word not in contractions else contractions[word] for word in text.split())
    text_expanded_contractions = text
    for abreviation in contraction_mapping:
        if text.find(abreviation) != -1:
            text_expanded_contractions = text_expanded_contractions.replace(abreviation, contraction_mapping[abreviation])
    return text_expanded_contractions


def remove_accented_chars(text):
    # Normal form D (NFD) is also known as canonical decomposition
    # It means that descompose á in (a + ´)  
    # Because ' belongs to category Mn it do not get into the final string.
    text_without_accent = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text_without_accent
    


def remove_special_chars(text, remove_digits=False):
    if remove_digits:
        pattern = r'[^a-zA ]'
    else:
        pattern = r'[^a-zA-Z0-9 ]'
    removed_special_chars = re.sub(pattern, "", text)    
    return removed_special_chars


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    text_without_stop_words = " ".join(word for word in tokenizer.tokenize(text) if not word.lower() in stopword_list)
    return text_without_stop_words


def remove_extra_new_lines(text):
    text_no_extra_lines = text.replace("\n"," ")
    return text_no_extra_lines


def remove_extra_whitespace(text):
    text_no_extra_whitespace = re.sub(' +', ' ', text)
    return text_no_extra_whitespace
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in tqdm(corpus):
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
