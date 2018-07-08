

#===========Text Classification========
import random
import nltk
from nltk.corpus import movie_reviews

docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

#doc = []

#for category in movie_reviews.categories():
#    for fileid in movie_reviews.fileids(category):
#        doc.append(list(movie_reviews.words(fileid), category))
#==============positive or negative feature===================
random.shuffle(docs)
print(docs[3])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#-----------------top commond words in review================    
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))    

#============number of words in list=========    
print(all_words["Kiran"])

#================features of words==============
word_features = list(all_words.keys())[:3000]

def find_features(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features 

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresset = [(find_features(rev), category) for (rev, category) in docs]  

#=================Naive bayes in NLP============
training_set = featuresset[:1900]
test_set = featuresset[1900:]

model1 = nltk.NaiveBayesClassifier.train(training_set)
print("Niave bayes accuracy:",(nltk.classify.accuracy(model1, test_set))*100)
model1.show_most_informative_features(15)

#===========Corpus==Wordnet==similarity bet words========
from nltk.corpus import wordnet
syns = wordnet.synsets ("program")

#sysnet
print(syns[0].name())
#just the word
print(syns[0].lemmas()[0].name())
#defination
print(syns[0].definition())
#examples
print(syns[0].examples())

#find synonyms
sysnonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        sysnonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            
print(set(sysnonyms))
print(set(antonyms))
            
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2))
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("air.n.01")

print(w1.wup_similarity(w2))
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")

print(w1.wup_similarity(w2))



#=============to find the path of file==========
import nltk
print(nltk.__file__)



#===================Lemmatizing==correct he miss spelled words==or co
#====convert to adjective or verb or noun form as per condition============
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))

print(lemmatizer.lemmatize("better", pos='a'))
print(lemmatizer.lemmatize("gaosse"))

#===================Named Entity Recognition==Same words group under one tag==============
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            nameEnt = nltk.ne_chunk(tagged, binary=True)
            
            nameEnt.draw()
            
    except Exception as e:
        print(str(e))
            
process_content() 

#===================Chinking================
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chink:{<.*>|}
            } <VB.? | IN |DT|TO>+{"""
            
            chunkParse = nltk.RegexpParser(chunkGram)
            chunked = chunkParse.parse(tagged)
                       
            #print(chunked)
            chunked.draw()
            
    except Exception as e:
        print(str(e))
            
process_content() 


#===============Chucking=Grouping========================
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunck:{<RB.?>*<VB.?>*<NNP>+<NN>?} """
            
            chunkParse = nltk.RegexpParser(chunkGram)
            chunked = chunkParse.parse(tagged)
                       
            #print(chunked)
            chunked.draw()
            
    except Exception as e:
        print(str(e))
            
process_content()        


#=============part of speech tagging=============
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
            
process_content()        

#=================Steming words===========
from nltk.stem import PorterStemmer 

ps = PorterStemmer()

expmle = ["Pythonli", "Python", "Pythoner", "Pythonala"]

for w in expmle:
    print(ps.stem(w))
#==========Stop words filteraton=================================
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

ex_text = "Hello Kiran, How are you doing today? The wether is great and sky is"

stop_words = set(stopwords.words("english")) 
print(stop_words)   

words = word_tokenize(ex_text)
filtersen = []

for w in words:
    if w not in stop_words:
        filtersen.append(w)
        
print(filtersen)



#=============================================
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download()

ex_text = "Hello Kiran, How are you doing today? The wether is great and sky is"

print(sent_tokenize(ex_text))
words = word_tokenize(ex_text)
print(words)

for i in words:
    print(i)
    
