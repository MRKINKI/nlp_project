# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import json
from utils.feature import FeatureExtract
from .question2text import Question2text
import collections


class Predict:
    def __init__(self, data_path, output_path, vocab, model):
        self.data_set = self.load_data(data_path)
        self.vocab = vocab
        self.output_path = output_path
        self.feature_extract = FeatureExtract()
        self.qt = Question2text()
        self.model = model

    def load_data(self, data_path):
        prepro_samples = []
        with open(data_path, encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                sample = json.loads(line.strip())
                prepro_samples.append(sample)
        return prepro_samples
        
    def non_find_answer(self):
        all_sample_num = 0
        non_answer_num = 0
        type_dict = {}
        for sample in self.data_set:
            for q in sample['questions']:
                all_sample_num += 1
                q_type = q['question_type']
                if q['find_answer'] == 0:
                    non_answer_num += 1
                    type_dict[q_type] = type_dict.get(q_type, 0) + 1 
        print(type_dict)
        print(all_sample_num, non_answer_num)
        print(non_answer_num/all_sample_num)
        
    def convert_to_ids(self, sample):
        sample['question_word_ids'] = self.vocab.word_vocab.convert_to_ids(sample['question_word'])
        sample['context_word_ids'] = self.vocab.word_vocab.convert_to_ids(sample['context_word'])
        sample['context_ner_ids'] = self.vocab.ner_vocab.convert_to_ids(sample['context_ner'])
        sample['context_pos_ids'] = self.vocab.pos_vocab.convert_to_ids(sample['context_pos'])
        return sample
        
    def get_batch(self, sample):
        batch = []
        chunks_tokens = sample['paragraphs_tokens']
        for q in sample['questions']:
            if q['bad_sample'] == 0:
                q_tokens = q['question_tokens']
                mrc = self.qt.find_best_question_match(chunks_tokens, q_tokens)
                q['most_related_para'] = mrc
                qa_sample_feature = self.feature_extract.extract(q, predict=True)
                input_sample = self.convert_to_ids(qa_sample_feature)
                batch.append(input_sample)
            else:
                print(q)
        return batch

    def predict(self, sample):
        batch = self.get_batch(sample)
        predictions, scores = self.model.predict(batch)
        assert len(predictions) == len([sam for sam in sample['questions'] if sam['bad_sample'] == 0]), \
            'HAS UNANSWERED SAMPLE'
        bad_sample_num = 0
        for idx, q in enumerate(sample['questions']):
            if q['bad_sample'] == 0:
                pred_idx = idx - bad_sample_num
                q['pred'], q['pred_score'] = predictions[pred_idx], float(scores[pred_idx])
            else:
                q['pred'], q['pred_score'] = '', 0
                bad_sample_num += 1
                
    def predict_formal(self):
        formal_result = []
        for idx, sample in enumerate(self.data_set):
            if sample['questions']:
                self.predict(sample)
            result = collections.OrderedDict()
            result['article_id'] = sample['article_id']
            result['questions'] = []
            for q in sample['questions']:
                sub_question = collections.OrderedDict()
                sub_question['questions_id'] = q['questions_id']
                sub_question['answer'] = q['pred']
                result['questions'].append(sub_question)
            formal_result.append(result)
            if idx%1000 == 0:
                print(idx)
        json.dump(formal_result, open('./data/deep/all_predict.json', 'w', encoding='utf-8'))

    def run(self):
        result = []
        for idx, sample in enumerate(self.data_set):
            if sample['questions']:
                self.predict(sample)
            result.append(sample)
            if idx % 1000 == 0:
                print(idx)
        json.dump(result, open(self.output_path, 'w', encoding='utf-8'))
        
    def get_format_json(self):
        result = []
        samples = json.load(open(self.output_path, encoding='utf-8'))
        for sample in samples:
            for q in sample['questions']:
                result.append({'true_answer': q['answer'], 
                               'extract_answer': q['pred']})
        json.dump(result, open('./data/deep/format_predict.json', 'w', encoding='utf-8'))
