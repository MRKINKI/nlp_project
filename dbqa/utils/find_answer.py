# -*- coding: utf-8 -*-

from collections import Counter

def precision_recall_f1(prediction, ground_truth):
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


def recall(prediction, ground_truth):
    return precision_recall_f1(prediction, ground_truth)[1]


def f1_score(prediction, ground_truth):
    return precision_recall_f1(prediction, ground_truth)[2]

def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    score = metric_fn(prediction, ground_truth)
    return score

def find_answer(paragraphs, qa_sample, maxlen):
    most_related_para_idx = -1
    most_related_para_len = 999999
    max_related_score = 0
    answer_words = qa_sample['answer_tokens']['cws']
    paragraph_words = [p['cws'] for p in paragraphs]
    for p_idx, para_words in enumerate(paragraph_words):
        if len(answer_words) > 0:
            related_score = metric_max_over_ground_truths(recall,
                                                          para_words,
                                                          answer_words)
        else:
            continue
        if related_score > max_related_score \
                or (related_score == max_related_score
                    and len(para_words) < most_related_para_len):
            most_related_para_idx = p_idx
            most_related_para_len = len(para_words)
            max_related_score = related_score
    qa_sample['most_related_para_idx'] = most_related_para_idx
    
    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    #answer_tokens = set()
    answer_words_set = set(answer_words)
    if qa_sample['most_related_para_idx'] != -1:
        most_related_para_words = paragraph_words[most_related_para_idx][:maxlen]
        for start_tidx in range(len(most_related_para_words)):
            if most_related_para_words[start_tidx] not in answer_words_set:
                continue
            for end_tidx in range(len(most_related_para_words) - 1, start_tidx - 1, -1):
                span_words = most_related_para_words[start_tidx: end_tidx + 1]
                if len(answer_words) > 0:
                    match_score = metric_max_over_ground_truths(f1_score, span_words,
                                                                answer_words)
                else:
                    match_score = 0
                if match_score == 0:
                    break
                if match_score > best_match_score:
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_answer = ''.join(span_words)
    if best_match_score > 0:
        qa_sample['answer_docs'] = best_match_d_idx
        qa_sample['answer_spans'] = best_match_span
        qa_sample['match_answer'] = best_answer
        qa_sample['match_score'] = best_match_score
        qa_sample['find_answer'] = 1
        qa_sample['most_related_para'] = paragraphs[most_related_para_idx]
    else:
        qa_sample['find_answer'] = 0

        
def get_sample(sample, maxlen):
    #samples = []
    #paragraphs = [p['cws'] for p in sample['paragraphs_tokens']]
    paragraphs = sample['paragraphs_tokens']
    for qa_sample in sample['questions']:
        if qa_sample['bad_sample'] == 0:
            find_answer(paragraphs, qa_sample, maxlen)
        #samples.append(new_sample)
    return sample        
        


