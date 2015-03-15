from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

def negate_words(mystring):

    numProcessed = 0
    mynegation=[]
    
    entry={}
    entry['words']=mystring.split()

    modifier = None
    negativeTerritory = 0

    for j in range(len(entry["words"])):
        word = entry["words"][j]
        if word in ["not", "n't","hardly",'don\'t','didn\'t','wouldn\'t','shouldn\'t','can\'t','couldn\'t',\
                   'haven\'t','hasn\'t','won\'t','never']:
            modifier = "vrbAdj"
            negativeTerritory = 2
        elif word in ["no", "none"]:
            modifier = "nouns"
            negativeTerritory = 2
        else:
            if negativeTerritory > 0:
                
                entry["words"][j] = "NEG" + word
                negativeTerritory -= 1
                if word[-1] in "?:!.,;-_`~+=#()/\|][*' ":negativeTerritory=0
                
    mynegation.append({'words':entry["words"]})
    numProcessed += 1
    return  ' '.join(mynegation[0]['words'])

def StripPunctuation(s):
    words=[]
    for word in s:
        exclude="?:!.,;-_`~+=#()/\|][*' "
        word = ''.join(ch if ch not in exclude else "" for ch in word )
        words+=word.split()
    return words

stopWords = stopwords.words("english")
stopWords=set(stopWords+["NEG"+elt for elt in stopWords]+["NEG"])


def prepareWords(review): 
    review=negate_words(review)
    s = re.sub(r'<.*/>', ' .',review)  
    s=re.sub("[^a-zA-Z]", " ", s) 
    word_list=StripPunctuation(nltk.word_tokenize(s))

            
    s=[WordNetLemmatizer().lemmatize(elt)\
       for elt in word_list if not ((elt in stopWords) or (len(elt)<2)) ]
    if len(s)<9: 
        s+=['NULL_WORD']
        #print(" ".join( s ))
    return( " ".join( s )) 

