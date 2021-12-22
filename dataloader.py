"""
Title: dataloader

Author: Rin
"""
import spacy
import numpy as np
import json
import pandas

from karazhan.uruloki.parser_trial import *
import tensorflow.keras as keras

def load_setup(file):
    f = open(file, 'r')
    content = f.read() # open the setup file
    settings = json.loads(content) 
    f.close() # close the setup file
    return settings

path = "C:/Users/doudou/Desktop/Always/MathQA/"

class Dataloader:
    def __init__(self,path,size = 100):
        self.size = size
        self.path= path
        self.train = load_setup(path+"train.json")
        self.test = load_setup(path+"test.json")
        self.corpus = []
        # Start the English Tokenizer
        self.tokenizer_en = keras.preprocessing.text.Tokenizer(10000)
        self.train_set = []
        
        self.collect_corpus()
        self.generate_pairs()
        
    def collect_corpus(self):
        corpus = []
        for i in range(self.size):
            corpus.append(self.train[i]["Problem"])
            corpus.append(self.test[i]["Problem"])
        self.corpus = corpus
        self.tokenizer_en.fit_on_texts(self.corpus)
        
        return corpus
    def update_corpus(self,texts):
        for i in range(len(texts)):
            self.corpus.append(texts[i])
        self.tokenizer_en.fit_on_texts(self.corpus)
        
    def generate_pairs(self):
        #generate a training set of pairs of [ X, program ]
        train_set = []
        for i in range(self.size):
            text = self.train[i]["Problem"]
            program = self.train[i]["annotated_formula"]
            pair_data = [self.tokenizer_en.texts_to_sequences([text]),Decompose(program)]
            train_set.append(pair_data)
        self.train_set = train_set
        self.test_set = []
    
    def convert_to_data(self,sentence):
        origin = self.tokenizer_en.texts_to_sequences([sentence])
        return origin
    
data = Dataloader(path)


tasks = [["what is the name of from the query", "get_name(query)"],
         ["give the name of the query","get_name(query)"],
         ["give me the name from the query","get_name(query)"],
         ["what is the name in the query","get_name(query)"],
         ["name of the query is","get_name(query)"],
         ["what is the index of name","get_index(get_name(query))"],
         ["what is the name index","get_index(get_name(query))"],
         ["give me the index of the name","get_index(get_name(query))"],
         ["give index of the name","get_index(get_name(query))"],
         ["what is the index of the name","get_index(get_name(query))"],
         ["change the name to negative","change_state(get_index(get_name(query)),Negative)"],
         ["set the name to negative","change_state(get_index(get_name(query)),Negative)"],
         ["name is the negative","change_state(get_index(get_name(query)),Negative)"],
         ["name is the positive","change_state(get_index(get_name(query)),Positive)"],
         ["set name to negative state","change_state(get_index(get_name(query)),Negative)"],
         ["set name to positive state","change_state(get_index(get_name(query)),Positive)"],
         ["name is negative","change_state(get_index(get_name(query)),Negative)"],
         ["name is positive","change_state(get_index(get_name(query)),Positive)"],
         ["name is in positive state","change_state(get_index(get_name(query)),Positive)"],
         ["name is in negative state","change_state(get_index(get_name(query)),Negative)"],
         ["name is negative state","change_state(get_index(get_name(query)),Negative)"],
         ["name is positive state","change_state(get_index(get_name(query)),Positive)"]
         ]

extras = ["open the folder","open the content appeared in the sentence","download the content",
          "composite the operator A and operator B","compose operators like A and B","Execture the code 2",
          "Exectute the code 2","Order 66 will be done", "release the cataclysm","Execute the first order",
          "Appear in the terminal","give the name from the query"]

for i in range(len(tasks)):
    extras.append([tasks[i][0]])

data.update_corpus(extras)


