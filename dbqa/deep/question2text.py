# -*- coding: utf-8 -*-
from collections import Counter
import datetime 
from .sim import F1, Alignment


def deco(func):
    def wrapper(*args, **kwargs):
        startTime = datetime.datetime.now()
        func(*args, **kwargs)
        endTime = datetime.datetime.now()
        cost_time = endTime - startTime
        msec = cost_time.total_seconds()*1000
        print("time is %f ms" %msec)
    return wrapper


class Question2text:
    def __init__(self):
        self.f1 = F1()
        self.alignment = Alignment()
    
    # @deco
    def find_best_question_match(self, 
                                 all_chunk_tokens, 
                                 question_tokens, 
                                 with_score=False, 
                                 method='f1',
                                 topk=5):
        most_related_chunk_idx = -1
        max_related_score = 0
        most_related_chunk_len = 0
        question_words = question_tokens['cws']
        chunk_score = dict()
        for p_idx, chunk_tokens in enumerate(all_chunk_tokens):
            chunk_words = chunk_tokens['cws']
            if len(question_words) > 0:
                related_score = self.get_relate_score(chunk_words, question_words, method=method)
            else:
                related_score = 0
            chunk_score[p_idx] = related_score
            #print(related_score)
            if related_score > max_related_score \
                    or (related_score == max_related_score \
                    and len(chunk_words) < most_related_chunk_len):
                most_related_chunk_idx = p_idx
                max_related_score = related_score
                most_related_chunk_len = len(chunk_words)
        if most_related_chunk_idx == -1:
            most_related_chunk_idx = 0
        sorted_chunk_score = sorted(chunk_score.items(), key=lambda f: f[1], reverse=True)
        return [all_chunk_tokens[chunk_info[0]] for chunk_info in sorted_chunk_score[:topk]]
        # if with_score:
        #     return most_related_chunk_idx, max_related_score
        # return all_chunk_tokens[most_related_chunk_idx]

    def get_relate_score(self, chunk_words, question_words, method='f1'):
        if method == 'f1':
            return self.f1.metric_max_over_ground_truths(self.f1.recall,
                                                         chunk_words,
                                                         question_words)
        elif method == 'align':
            _, _, max_score = self.alignment.waterman(chunk_words, question_words)
            return max_score
