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

def find_answer(paragraphs, sample):
    most_related_para_idx = -1
    most_related_para_len = 999999
    max_related_score = 0
    answer_tokens = sample['answer_token']
    for p_idx, para_tokens in enumerate(paragraphs):
        if len(answer_tokens) > 0:
            related_score = metric_max_over_ground_truths(recall,
                                                          para_tokens,
                                                          answer_tokens)
        else:
            continue
        if related_score > max_related_score \
                or (related_score == max_related_score
                    and len(para_tokens) < most_related_para_len):
            most_related_para_idx = p_idx
            most_related_para_len = len(para_tokens)
            max_related_score = related_score
    sample['most_related_para_idx'] = most_related_para_idx
    
    best_match_score = 0
    best_match_d_idx, best_match_span = -1, [-1, -1]
    best_fake_answer = None
    #answer_tokens = set()
    answer_tokens_set = set(answer_tokens)
    if sample['most_related_para_idx'] != -1:
        most_related_para_tokens = paragraphs[most_related_para_idx][:500]
        for start_tidx in range(len(most_related_para_tokens)):
            if most_related_para_tokens[start_tidx] not in answer_tokens_set:
                continue
            for end_tidx in range(len(most_related_para_tokens) - 1, start_tidx - 1, -1):
                span_tokens = most_related_para_tokens[start_tidx: end_tidx + 1]
                if len(answer_tokens) > 0:
                    match_score = metric_max_over_ground_truths(f1_score, span_tokens,
                                                                answer_tokens)
                else:
                    match_score = 0
                if match_score == 0:
                    break
                if match_score > best_match_score:
                    best_match_span = [start_tidx, end_tidx]
                    best_match_score = match_score
                    best_answer = ''.join(span_tokens)
    if best_match_score > 0:
        sample['answer_docs'] = best_match_d_idx
        sample['answer_spans'] = best_match_span
        sample['match_answer'] = best_answer
        sample['match_score'] = best_match_score
        sample['find_answer'] = 1
        sample['most_related_para'] = paragraphs[most_related_para_idx]
    else:
        sample['find_answer'] = 0
    return sample
        
def get_sample(sample):
    samples = []
    paragraphs = sample['paragraphs_token']
    for question_doc in sample['questions']:
        new_sample = find_answer(paragraphs, question_doc)
        samples.append(new_sample)
    return samples        
        


