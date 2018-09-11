# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import collections
from collections import Counter
import string
import torch
import random


class BatchGen:
    def __init__(self, data,opt ,batch_size, gpu, evaluation=False):
        '''
        input:
            data - list of lists
            batch_size - int
        '''
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu
        self.opt = opt
        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 6
            else:
                assert len(batch) == 8

            context_len = max(len(x) for x in batch[0])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[1][0][0])
            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.opt['pos_size']).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1
                        
            context_ent = torch.Tensor(batch_size, context_len, self.opt['ner_size']).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1
#                context_ent[i, :len(doc)] = torch.LongTensor(doc)
                
            question_len = max(len(x) for x in batch[4])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[4]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            if not self.eval:
                y_s = torch.LongTensor([int(t) for t in batch[5]])
                y_e = torch.LongTensor([int(t) for t in batch[6]])
            text = list(batch[-1])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text)

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    '''
    计算EM
    '''
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False

def _f1_score(pred, answers):
    '''
    计算F1
    '''
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    #g_tokens = _normalize_answer(pred).split()
    g_tokens = [t for t in _normalize_answer(pred)]
    #scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    scores = [_score(g_tokens, [t for t in _normalize_answer(a)]) for a in answers]
    return max(scores)

def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1   
