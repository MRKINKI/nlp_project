# -*- coding: utf-8 -*-

import json
from doc_reader.drqa_model import DrqaModel
import os
import pickle

with open('./data/cetc/test.json', encoding='utf-8') as fin:
    for line in fin:
        sam = json.loads(line.strip())
        break
    
class predict:
    def __init__(self, data_path, output_path):
        self.data_set = self.load_data(data_path)
        self.output_path = output_path
        
        
    def load_data(self, data_path):
        prepro_samples = []
        with open(data_path, encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                sample = json.loads(line.strip())
                prepro_samples.append(sample)
        return prepro_samples
        
    def predict(self, sample):
        result = []
        chunks = sample['sentences_tokens']
        for q in sample['questions']:
            if q['bad_sample'] == 0:
                q_tokens = q['question_tokens']
                q_type = q['question_type']
                a_tokens = q['answer_tokens']
                mrc = self.qt.find_best_question_match(chunks, q_tokens)
            else:
                print(q)
            try:
                answer = self.answer_extract.extract(mrc, q_tokens)
            except:
                answer = sample['article_title']
            result.append({'q':' '.join(q_tokens['cws']), 
                           'mrc':' '.join(mrc['cws']), 
                           'true_answer': ''.join(a_tokens['cws']), 
                           'extract_answer': answer,
                           'q_type': q_type})
        return result
    
if __name__ == '__main__':
    vocab_dir = './data/cetc/'
    data_path = './data/cetc/test.json'
    output_path = './data/deep/origin_predict.json'
    with open(os.path.join(vocab_dir, 'vocab.data'), 'rb') as fin:
        data_vocabs = pickle.load(fin)
    rc_model = DrqaModel(data_vocabs.word_vocab, eva=True)
    pred = predict(data_path, output_path)