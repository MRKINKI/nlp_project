# -*- coding: utf-8 -*-
import json
from .question2text import Question2text
from .answer_extract import AnswerExtract


class Predict:
    def __init__(self, data_path, output_path):
        self.answer_extract = AnswerExtract('waterman')
        self.qt = Question2text()
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
    
    def predict_formal(self, sample):
        result = {}
        chunks = sample['sentences_token']
        result['article_id'] = sample['article_id']
        result['questions'] = []
        for q in sample['questions']:
            sub_question = dict()
            sub_question['questions_id'] = q['questions_id']
            q_tokens = q['question_token']
            # q_type = q['question_type']
            mrc = self.qt.find_best_question_match(chunks, q_tokens)
            try:
                answer = self.answer_extract.extract(mrc, q_tokens)
            except:
                answer = sample['article_title']
            sub_question['answer'] = answer
            result['questions'].append(sub_question)
        return result
            
        
    def run(self, method='formal'):
        all_result = []
        for idx,sample in enumerate(self.data_set):
            if method == 'formal':
                all_result.append(self.predict_formal(sample))
            else:
                all_result.extend(self.predict(sample))
            if idx%1000 == 0:
                print(idx)
        json.dump(all_result, open(self.output_path, 'w', encoding='utf-8'))
