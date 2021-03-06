# -*- coding: utf-8 -*-

from utils.prepro import prepro_token
import json
import sys
from trad.predict import Predict
sys.path.append('../nlp')
from ltpnlp import LtpNlp

raw_data_path = './data/cetc/question.json'
prepro_data_path = './data/trad/all.json'
formal_predict_data_path = './data/trad/all_predict.json'
predict_data_path = './data/trad/local_all_predict.json'

def preprocess():
    ln = LtpNlp()
    tokenizer = ln.tokenize
    prepro_token(raw_data_path, 
                 prepro_data_path, 
                 tokenizer, 
                 extract_sample=False, 
                 chunk='sentences')    
    ln.release()

#preprocess()


#pre = Predict(prepro_data_path, predict_data_path)
#pre.run(method='unformal')
#
#a = json.load(open(predict_data_path, encoding='utf-8'))[:1000]


#with open(prepro_data_path, encoding='utf-8') as fin:
#    for line in fin:
#        sam = json.loads(line.strip())
#        break



#preds = json.load(open('./data/pred.json'))

#c = preds[:1000]