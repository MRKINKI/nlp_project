# -*- coding: utf-8 -*-

import sys
import json
import re
from .find_answer import get_sample
import numpy as np


def tokenize_dict(tokenizer, sample, fields, module='ner'):
    bad_sample = 0
    for field in fields:
        data = sample[field]
        if isinstance(data, str):
            if data.strip():
                token_dict = tokenizer(data, module)
                del token_dict['text']
                token_data = token_dict
            else:
                bad_sample = 1
        elif isinstance(data, list):
            token_data = []
            for d in data:
                token_dict = tokenizer(d, module)
                del token_dict['text']
                token_data.append(token_dict)
        if bad_sample == 0:
            sample[field + '_tokens'] = token_data
        if 'answer' in sample:
            if 'bad_sample' not in sample:
                sample['bad_sample'] = bad_sample
            elif sample['bad_sample'] == 0:
                sample['bad_sample'] = bad_sample

def get_paragraphs(sample):      
    paragraphs = []
    paragraphs.append(sample['article_title'])
    for paragraph in re.split('　　|  ', sample['article_content']):
        para = paragraph.strip()
        if para:
            paragraphs.append(para)
    sample['paragraphs'] = paragraphs 
    
def get_sentences(sample):
    sentences = []
    sentences.append(sample['article_title'])
    content = sample['article_content']
    for sentence in re.split('。|！|？|', content):
        sentence = sentence.strip()
        if sentence:
            sentences.append(sentence)
    sample['sentences'] = sentences
    
    
def build_samples(sample, maxlen):
    return get_sample(sample, maxlen)
    

def prepro_token(infile, outfile, tokenizer, extract_sample=False, chunk='sentences', maxlen=500):
    origin_data = json.load(open(infile, encoding='utf-8'))
    with open(outfile, 'w', encoding='utf-8') as fout:
        for idx, sample in enumerate(origin_data):
#            if idx < 13017:
#                continue
            if idx % 1 == 0:
                print(idx)
            if chunk == 'paragraphs':
                get_paragraphs(sample)
            elif chunk == 'sentences':
                get_sentences(sample)
            tokenize_dict(tokenizer, sample, [chunk])
            for question in sample['questions']:
                tokenize_dict(tokenizer, question, ['question'])
                #tokenize_dict(tokenizer, question, ['answer', 'question'])
            if extract_sample:
                sample = build_samples(sample, maxlen)
#                new_samples = build_samples(sample)
#                for n_sample in new_samples:
#                    fout.write(json.dumps(n_sample, ensure_ascii=False)+'\n')
            fout.write(json.dumps(sample, ensure_ascii=False)+'\n')
#            if idx > 500:
#                break

def save_file(data, file):
    with open(file, 'w', encoding='utf-8') as fout:
        for sample in data:
            fout.write(json.dumps(sample, ensure_ascii=False)+'\n')
            

def train_test_split(all_data_file, trainfile, testfile, train_rate):
    all_samples = []
    with open(all_data_file, encoding='utf-8') as fin:
        for idx, line in enumerate(fin):
            sample = json.loads(line.strip())
            all_samples.append(sample)
    # print(len(all_samples))
    np.random.shuffle(all_samples)
    size = len(all_samples)
    train_data = all_samples[:int(size*train_rate)]
    test_data = all_samples[int(size*train_rate):]
    save_file(train_data, trainfile)
    save_file(test_data, testfile)          

if __name__ == '__main__':
    pass
    train_rate = 0.8
    train_file = '../data/cetc/train.json'
    test_file = '../data/cetc/test.json'
    infile = '../data/cetc/question.json'
    outfile = '../data/cetc/para_data.json'
#    train_test_split(outfile, train_file, test_file, train_rate)
    ln = LtpNlp()
    tokenizer = ln.tokenize
    prepro(infile, outfile, tokenizer, extract_sample=True, chunk='paragraphs')
