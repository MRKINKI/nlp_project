# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import json
from utils.feature import FeatureExtract
from .question2text import Question2text
import collections
from doc_reader.drqa_model import DrqaModel
import os
import pickle
import numpy as np
import msgpack
import copy
import math
from rule.main import Rule


class Predict:
    def __init__(self, data_path, output_path, vocab, model):
        self.data_set = self.load_data(data_path)
        self.vocab = vocab
        self.output_path = output_path
        self.feature_extract = FeatureExtract()
        self.qt = Question2text()
        self.model = model
        self.ru = Rule()

    def load_data(self, data_path):
        prepro_samples = []
        with open(data_path, encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                sample = json.loads(line.strip())
                for qa_sample in sample['questions']:
                    if 'bad_sample' not in qa_sample:
                        if qa_sample['question'].strip():
                            qa_sample['bad_sample'] = 0
                        else:
                            qa_sample['bad_sample'] = 1
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

    def cut_mrc(self, mrc):
        cut_mrcs = []
        # cut_mrcs = [mrc]
        num = math.ceil(len(mrc['cws']) / 1000)
        for i in range(num):
            sub_mrc = {'cws': mrc['cws'][1000*i: 1000*(i+1)],
                       'pos': mrc['pos'][1000*i: 1000*(i+1)],
                       'ner': mrc['ner'][1000*i: 1000*(i+1)]}
            cut_mrcs.append(sub_mrc)
        return cut_mrcs

    def get_batch(self, sample, method, topk, find_question_match):
        batch = []
        chunks_tokens = sample['paragraphs_tokens']
        q_ids = []
        all_mrcs = []
        for q in sample['questions']:
            if q['bad_sample'] == 0:
                q_id = q['questions_id']
                q_tokens = q['question_tokens']
                if find_question_match:
                    mrcs = self.qt.find_best_question_match(chunks_tokens, q_tokens, method=method, topk=topk)
                else:
                    mrcs = [q['most_related_para']]
                for mrc in mrcs:
                    cut_mrcs = self.cut_mrc(mrc)
                    for cut_mrc in cut_mrcs:
                        q['most_related_para'] = cut_mrc
                        qa_sample_feature = self.feature_extract.extract(q, predict=True)
                        input_sample = self.convert_to_ids(qa_sample_feature)
                        q_ids.append(q_id)
                        batch.append(input_sample)
                        all_mrcs.append(cut_mrc)
            else:
                print(q)
        return batch, q_ids, all_mrcs

    def get_lit_pro_batch(self, sample, method, topk, alpha):
        batch = []
        chunks_tokens = sample['paragraphs_tokens']
        q_ids = []
        all_mrcs = []
        for q in sample['questions']:
            if q['bad_sample'] == 0:
                if q['pred_score'] < alpha:
                    q_id = q['questions_id']
                    q_tokens = q['question_tokens']
                    mrcs = self.qt.find_best_question_match(chunks_tokens, q_tokens, method=method, topk=topk)
                    for mrc in mrcs:
                        q['most_related_para'] = mrc
                        qa_sample_feature = self.feature_extract.extract(q, predict=True)
                        input_sample = self.convert_to_ids(qa_sample_feature)
                        q_ids.append(q_id)
                        batch.append(input_sample)
                        all_mrcs.append(mrc)
        return batch, q_ids, all_mrcs

    def write2sample(self, predictions, scores, score_matrices, q_ids, sample, all_mrcs):
        pred2id = dict()
        for pred_idx, q_id in enumerate(q_ids):
            if q_id not in pred2id:
                pred2id[q_id] = pred_idx
            elif scores[pred_idx] > scores[pred2id[q_id]]:
                pred2id[q_id] = pred_idx

        for idx, q in enumerate(sample['questions']):
            q_id = q['questions_id']
            if q_id in pred2id:
                pred_idx = pred2id[q_id]
                q['pred'], q['pred_score'] = predictions[pred_idx], float(scores[pred_idx])
                s_matrix = score_matrices[pred_idx]
                a_mrc = all_mrcs[pred_idx]
                mrc_len = len(a_mrc['cws'])
                q['score_matrix'] = s_matrix[:mrc_len, :mrc_len]
                q['most_related_para'] = a_mrc
            elif 'pred' not in q:
                q['pred'], q['pred_score'], q['score_matrix'] = '', 0, []

    def predict(self, sample, method='align', topk=1, find_question_match=True):
        batch, q_ids, all_mrcs = self.get_batch(sample, method, topk=topk, find_question_match=find_question_match)
        predictions, scores, score_matrices = self.model.predict(batch)
        self.write2sample(predictions, scores, score_matrices, q_ids, sample, all_mrcs)
        # batch, q_ids = self.get_lit_pro_batch(sample, method, topk=5, alpha=0.1)
        # if batch:
        #     predictions, scores, score_matrices = self.model.predict(batch)
        #     self.write2sample(predictions, scores, score_matrices, q_ids, sample)

    def clear(self, sample):
        for qa_sample in sample['questions']:
            qa_sample['score_matrix'] = []

    def ensemble(self, model_infos, output_path):

        for idx, model_info in enumerate(model_infos):
            model_adr = model_info[0]
            args_file = os.path.join(model_adr, 'args.pkl')
            args = pickle.load(open(args_file, 'rb'))
            with open(os.path.join(model_adr, 'vocab.data'), 'rb') as fin:
                data_vocabs = pickle.load(fin)
            args.pos_size = data_vocabs.pos_vocab.size()
            args.ner_size = data_vocabs.ner_vocab.size()
            args.resume_file = model_info[1]
            self.vocab = data_vocabs
            self.model = DrqaModel(data_vocabs.word_vocab,
                                   args,
                                   eva=True)
            find_question_match = True if idx == 0 else False

            # sub_output_path = os.path.join(output_path, str(idx))
            # if not os.path.exists(sub_output_path):
            #     os.makedirs(sub_output_path)

            for jdx, sample in enumerate(self.data_set):
                # if jdx < 13000:
                #     continue

                if sample['questions']:
                    self.predict(sample, find_question_match=find_question_match)
                # score_sample = copy.copy(sample)
                # result.append(sample)
                # self.clear(sample)
                if (jdx+1) % 1000 == 0:
                    print(jdx)
                    # output_file = os.path.join(sub_output_path, str(int((jdx+1) / 1000)))
                    output_file = os.path.join(output_path, str(int((jdx+1) / 1000)))
                    if os.path.exists(output_file):
                        history_data = pickle.load(open(output_file, 'rb'))
                        for hdx, sample in enumerate(history_data):
                            for hjdx, qa_sample in enumerate(sample['questions']):
                                self.data_set[jdx - 999 + hdx]['questions'][hjdx]['score_matrix'] += qa_sample['score_matrix']
                        del history_data

                    pickle.dump(self.data_set[jdx-999: jdx+1], open(output_file, 'wb'))
                    # msgpack.dump(self.data_set[jdx-999: jdx+1], open(output_file, 'wb'))
                    for s_idx in range(jdx-999, jdx+1):
                        self.clear(self.data_set[s_idx])
                    # print('clear')
                    # print(self.data_set[233]['questions'][0]['score_matrix'])

    def get_ensemble_result(self, ensemble_path, output_file, sample_num=20):
        all_data = []
        for i in range(1, sample_num+1):
            print(i)

            data = pickle.load(open(os.path.join(ensemble_path, str(i)), 'rb'))

            for sdx, sample in enumerate(data):
                qa_samples = sample['questions']
                for qa_sample in qa_samples:
                    if qa_sample['bad_sample']:
                        qa_sample['pred'] = ''
                        continue
                    # del qa_sample['score_matrix']
                    score_matrix = qa_sample['score_matrix']
                    # print(score_matrix)
                    text = qa_sample['most_related_para']['cws']
                    s_idx, e_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
                    score = np.max(score_matrix)
                    prediction = ''.join(text[s_idx:e_idx + 1])
                    del qa_sample['score_matrix']
                    qa_sample['pred_answer_span'] = [str(s_idx), str(e_idx)]
                    qa_sample['pred'] = prediction
                    qa_sample['score'] = str(score)
            all_data.extend(data)
        json.dump(all_data, open(output_file, 'w', encoding='utf-8'))

    def predict_formal_from_file(self, inputfile, outputfile):
        data = json.load(open(inputfile, encoding='utf-8'))
        formal_result = []
        for idx, sample in enumerate(data):
            self.ru.run(sample)
            result = collections.OrderedDict()
            result['article_id'] = sample['article_id']
            result['questions'] = []
            for q in sample['questions']:
                sub_question = collections.OrderedDict()
                sub_question['questions_id'] = q['questions_id']
                sub_question['answer'] = q['pred']
                result['questions'].append(sub_question)
            formal_result.append(result)
            if idx % 1000 == 0:
                print(idx)
        json.dump(formal_result, open(outputfile, 'w', encoding='utf-8'))

                
    def predict_formal(self, file):
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
            if idx % 1000 == 0:
                print(idx)
        json.dump(formal_result, open(file, 'w', encoding='utf-8'))

    def run(self, output_path):
        result = []
        for idx, sample in enumerate(self.data_set):
            if sample['questions']:
                self.predict(sample)
            result.append(sample)
            if idx % 1000 == 0:
                print(idx)
        json.dump(result, open(output_path, 'w', encoding='utf-8'))
        
    def get_format_json(self, output_path):
        result = []
        samples = json.load(open(output_path, encoding='utf-8'))
        for sample in samples:
            for q in sample['questions']:
                result.append({'true_answer': q['answer'], 
                               'extract_answer': q['pred']})
        json.dump(result, open('./data/deep/format_predict.json', 'w', encoding='utf-8'))
