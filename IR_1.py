import numpy as np 
import pandas as pd
import json 
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import requests


json_file = 'https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_01.json'
handle = requests.get(json_file)
season_1 = json.loads(handle.text) 

friends_data = [] ## the master list that will contain the complete dataset
## each element of the list will be the complete dictionary that contains that particular seasons dataset, indexed with 0 
## friends_data[0] is the season 1 dataset and so on 
friends_data.append(season_1)

for season_index in range(2, 11): ##merging the dataset of all seasons 
    season_index = '0%d'%season_index if season_index <10 else str(season_index)
    
    json_url = 'https://raw.githubusercontent.com/emorynlp/character-mining/master/json/friends_season_%s.json'%season_index
    request = requests.get(json_url)
    curr_season = json.loads(request.text) ## the current season 
    friends_data.append(curr_season)

## merging complete 

dialogue = {} ## dictionary that stores all the dialogues, grouped by scene id 
num_scenes = 0 ## total number of scenes in all the episodes 

for season in friends_data:
    episodes = season['episodes']
    for episode in episodes:
        scenes = episode['scenes']
        for scene in scenes:
            scene_id = scene['scene_id']
            num_scenes += 1 ## calculating the total number of scenes in the whole show
            utterances = scene['utterances']
            for utterance in utterances: 
                transcript = utterance['transcript']
                if scene_id not in dialogue.keys():
                    dialogue[scene_id] = transcript
                    dialogue[scene_id] += '\n'
                else:
                    dialogue[scene_id] += transcript
                    dialogue[scene_id] += '\n'

tokenizer = RegexpTokenizer("[\w']+")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 

docs = dict.fromkeys(dialogue.keys(), []) ## this dictionary contains normalized tokens grouped by scene id
for key in dialogue: 
    tokens = tokenizer.tokenize(dialogue[key])
    stopset = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if not w in stopset]
    stemmed_sentence = [stemmer.stem(w) for w in filtered_sentence]
    lemmatized_sentence = [lemmatizer.lemmatize(w) for w in stemmed_sentence]
    docs[key] = lemmatized_sentence

## creating the inverted index
## the keys are the tokens, the values are the scene is occurs in and the frequency in each scene
inverted_index = {} 
for key in docs:
    for word in docs[key]:
        if word not in inverted_index.keys():
            sc = {}
            sc[key] = 1
            inverted_index[word] = sc
        else:
            if key not in inverted_index[word].keys():
                inverted_index[word][key] = 1
            else:
                inverted_index[word][key] += 1       

query = input('Enter a dialogue:\n')
print("1.Natural Term Frequency\n2.Logarithm Term Frequency\n3.Augmented Term Frequency\n4.Boolean Term Frequency\n")
case = input("Choose your scoring scheme:\n")

## collecting the normalized tokens for the query
tokens = tokenizer.tokenize(query)
stopset = set(stopwords.words('english'))
filtered_sentence = [w for w in tokens if not w in stopset]
stemmed_sentence = [stemmer.stem(w) for w in filtered_sentence]
lemmatized_sentence = [lemmatizer.lemmatize(w) for w in stemmed_sentence]
query = lemmatized_sentence

def term_freq(term,doc): ## term frequency
    return inverted_index[term][doc]

def doc_freq(term): ## document frequency 
    return len(inverted_index[term])

def inverted_doc_freq(term): ##inverted document frequency
    return num_scenes/doc_freq(term)

def log_term_freq(term,doc): ##logarithmic term frequency
    return 1+math.log(term_freq(term,doc))

def aug_term_freq(term,doc,_max): ##augmented term frequency
     return 0.5+(0.5*term_freq(term,doc)/max_)
    
def bool_term_freq(term,doc): ## boolean term frequency 
    if term_freq(term,doc)>0:
        return 1
    else:
        return 0

max_=0
for term in query:
    if term not in inverted_index.keys():
        continue
    for key in inverted_index[term]:
        if term_freq(term, key)>max_:
            max_= term_freq(term,key)

def score(term,doc): ##function to return tf-idf score based on chosen scoring scheme
    if(int(case)==1):
        return term_freq(term,doc)*inverted_doc_freq(term)
    if(int(case)==2):
        return log_term_freq(term,doc)*inverted_doc_freq(term)
    if(int(case)==3):
        return aug_term_freq(term,doc,max_)*inverted_doc_freq(term)
    if(int(case)==4):
        return bool_term_freq(term,doc)*inverted_doc_freq(term)

score_matrix = {}
for term in query:
    if term not in inverted_index.keys():
        continue
    for key in inverted_index[term]:
        if key not in score_matrix:
            score_matrix[key] = []
            score_matrix[key].append(1)
            score_matrix[key].append(score(term, key))
        else:
            score_matrix[key][1] += score(term, key)
            score_matrix[key][0] +=1

sorted_indices = sorted(score_matrix.items(), key=lambda kv: kv[1], reverse=True)

print('These are the top 10 scenes related to the given query')
for i in range(10):
    print(str(i+1) + ". "+ sorted_indices[i][0] + " " + str(sorted_indices[i][1]))