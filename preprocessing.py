import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
stopword=list(stopwords.words('english'))

def clean_text(sent):
    return re.sub(r"[^\w\s]","",sent)

def lower_sentence(sent):
    return sent.lower()

def StopWords_Removal(sent):
    sent=sent.split()
    words=[s for s in sent if s not in stopword]
    return " ".join(words)

def lemmatizer_func(sent):
    sent=sent.split()
    lem_ans=[WordNetLemmatizer().lemmatize(word) for word in sent]
    return lem_ans

def stemmer_func(sent):
    sent=sent.split()
    stem_ans=[PorterStemmer().stem(word) for word in sent]
    return stem_ans

def word_tokenization(sent):
    return nltk.word_tokenize(sent)

def sent_tokenization(sent):
    return nltk.sent_tokenize(sent)


if __name__=="__main__":
    sent=input("Enter Sentence:-")
    sent=clean_text(sent)
    sent=lower_sentence(sent)
    sent=StopWords_Removal(sent)
    lem=lemmatizer_func(sent)
    stm=stemmer_func(sent)
    print("--------------------------------------")
    print(sent)
    print("--------------------------------------")
    print("Stemmer:-",stm)
    print("--------------------------------------")
    print("Lemmatizer:-",lem)