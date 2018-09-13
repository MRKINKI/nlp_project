# -*- coding: utf-8 -*-
from collections import Counter
import datetime 

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
        pass
    
    # @deco
    def find_best_question_match(self, all_chunk_tokens, question_tokens, with_score=False):
        most_related_chunk_idx = -1
        max_related_score = 0
        most_related_chunk_len = 0
        question_words = question_tokens['cws']
        for p_idx, chunk_tokens in enumerate(all_chunk_tokens):
            chunk_words = chunk_tokens['cws']
            if len(question_words) > 0:
                related_score = self.metric_max_over_ground_truths(self.recall,
                                                                   chunk_words,
                                                                   question_words)
            else:
                related_score = 0
    
            if related_score > max_related_score \
                    or (related_score == max_related_score \
                    and len(chunk_tokens) < most_related_chunk_len):
                most_related_chunk_idx = p_idx
                max_related_score = related_score
                most_related_chunk_len = len(chunk_tokens)
        if most_related_chunk_idx == -1:
            most_related_chunk_idx = 0
        if with_score:
            return most_related_chunk_idx, max_related_score
        return all_chunk_tokens[most_related_chunk_idx]
        
    def precision_recall_f1(self, prediction, ground_truth):
        if not isinstance(prediction, list):
            prediction_tokens = prediction.split()
        else:
            prediction_tokens = prediction
        if not isinstance(ground_truth, list):
            ground_truth_tokens = ground_truth.split()
        else:
            ground_truth_tokens = ground_truth
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        p = 1.0 * num_same / len(prediction_tokens)
        r = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * p * r) / (p + r)
        return p, r, f1

    def recall(self, prediction, ground_truth):
        return self.precision_recall_f1(prediction, ground_truth)[1]
    
    def f1_score(self, prediction, ground_truth):
        return self.precision_recall_f1(prediction, ground_truth)[2]
    
    def metric_max_over_ground_truths(self, metric_fn, prediction, ground_truth):
        score = metric_fn(prediction, ground_truth)
        return score