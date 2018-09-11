# -*- coding: utf-8 -*-

import json
import collections


class FeatureExtract:
    def __init__(self):
        pass

    def get_context_feature(self, question_words, context_words):
        counter_ = collections.Counter(w.lower() for w in context_words)
        total = sum(counter_.values())
        frequence_feature = [counter_[w.lower()] / total for w in context_words]
        extract_match_feature = [t in question_words for t in context_words]
        return frequence_feature, extract_match_feature
        
    def extract(self, sample):
        new_sample = {}
        question_words, context_words = sample['question_tokens']['cws'], \
                                        sample['most_related_para']['cws']
        freq, em = self.get_context_feature(question_words, context_words)
        new_sample['context_freq'] = freq
        new_sample['context_em'] = em
        new_sample['context_word'] = sample['most_related_para']['cws']
        new_sample['context_ner'] = sample['most_related_para']['ner']
        new_sample['context_pos'] = sample['most_related_para']['pos']
        new_sample['question_word'] = sample['question_tokens']['cws']
        new_sample['question_ner'] = sample['question_tokens']['ner']
        new_sample['question_pos'] = sample['question_tokens']['pos']
        new_sample['answer_spans'] = sample['answer_spans']
        new_sample['answer_start'] = sample['answer_spans'][0]
        new_sample['answer_end'] = sample['answer_spans'][1]
        return new_sample


if __name__ == '__main__':
    path = '../data/cetc/train.json'
    with open(path, encoding='utf-8') as fin:
        for line in fin:
            sam = json.loads(line.strip())
            break
    qa_sample = sam['questions'][0]
    q_words, c_words = qa_sample['question_tokens']['cws'], qa_sample['most_related_para']['cws']
    fea = FeatureExtract()
    a, b = fea.get_context_feature(q_words, c_words)
    sample = fea.extract(qa_sample)
