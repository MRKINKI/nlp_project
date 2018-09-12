# -*- coding: utf-8 -*-

import re
from collections import Counter
import string
import torch
import random


class BatchTransform:
    def __init__(self, args):
        self.opt = args.__dict__

    def transform(self, batch, eva=False):
        batch_size = len(batch)

        max_context_len = max(len(sample['context_word_ids']) for sample in batch)
        context_id = torch.LongTensor(batch_size, max_context_len).fill_(0)
        for i, sample in enumerate(batch):
            context_word_ids = sample['context_word_ids']
            context_id[i, :len(context_word_ids)] = torch.LongTensor(context_word_ids)

        feature_len = len(batch[0]['context_feature'][0])
        context_feature = torch.Tensor(batch_size, max_context_len, feature_len).fill_(0)
        for i, sample in enumerate(batch):
            context_feature_raw = sample['context_feature']
            for j, feature in enumerate(context_feature_raw):
                context_feature[i, j, :] = torch.Tensor(feature)

        context_pos = torch.Tensor(batch_size, max_context_len, self.opt['pos_size']).fill_(0)
        for i, sample in enumerate(batch):
            context_pos_raw = sample['context_pos_ids']
            for j, pos in enumerate(context_pos_raw):
                context_pos[i, j, pos] = 1

        context_ner = torch.Tensor(batch_size, max_context_len, self.opt['ner_size']).fill_(0)
        for i, sample in enumerate(batch):
            context_ner_raw = sample['context_ner_ids']
            for j, ner in enumerate(context_ner_raw):
                context_ner[i, j, ner] = 1
#                context_ent[i, :len(doc)] = torch.LongTensor(doc)

        max_question_len = max(len(sample['question_word_ids']) for sample in batch)
        question_id = torch.LongTensor(batch_size, max_question_len).fill_(0)
        for i, sample in enumerate(batch):
            question_word_ids = sample['question_word_ids']
            question_id[i, :len(question_word_ids)] = torch.LongTensor(question_word_ids)

        context_mask = torch.eq(context_id, 0)
        question_mask = torch.eq(question_id, 0)
        if not eva:
            y_s = torch.LongTensor([int(sample['start_id']) for sample in batch])
            y_e = torch.LongTensor([int(sample['end_id']) for sample in batch])
        text = [sample['context_word'] for sample in batch]
        if self.opt['cuda']:
            context_id = context_id.pin_memory()
            context_feature = context_feature.pin_memory()
            context_pos = context_pos.pin_memory()
            context_ner = context_ner.pin_memory()
            context_mask = context_mask.pin_memory()
            question_id = question_id.pin_memory()
            question_mask = question_mask.pin_memory()
        if eva:
            return (context_id, context_feature, context_pos, context_ner, context_mask,
                    question_id, question_mask, text)
        else:
            return (context_id, context_feature, context_pos, context_ner, context_mask,
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
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
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
