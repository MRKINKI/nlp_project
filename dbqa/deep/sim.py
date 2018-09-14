# -*- coding: utf-8 -*-
from collections import Counter

class Alignment:
    
    def __init__(self):
        self.match_award      = 100
        self.mismatch_penalty = -100
        self.gap_penalty      = -5 # both for opening and extanding

    def zeros(self, shape):
        retval = []
        for x in range(shape[0]):
            retval.append([])
            for y in range(shape[1]):
                retval[-1].append(0)
        return retval
    

    def match_score(self, alpha, beta):
        if alpha == beta:
            return self.match_award
    #    elif alpha == '-' or beta == '-':
    #        return gap_penalty
        else:
            return self.mismatch_penalty

    def lcs(self, seq1, seq2):
        m, n = len(seq1), len(seq2)
        inverse_lcs_idxes = []
        inverse_unmatch_idxes = []
        # 记录最大公共子序列长度
        score = self.zeros((m+1, n+1))
        # 记录最大公共子序列路径
        pointer = self.zeros((m+1, n+1))
        for i in range(1, m+1):
            for j in range(1, n+1):
                if seq1[i-1] == seq2[j-1]:
                    score[i][j] = score[i-1][j-1]+1
                    pointer[i][j] = '↖'
                elif score[i-1][j] > score[i][j-1]:
                    score[i][j] = score[i-1][j]
                    pointer[i][j] = '↑'
                else:
                    score[i][j] = score[i][j-1]
                    pointer[i][j] = '←'
        a_idx, b_idx = m, n
        sub_inverse_unmatch_idxes = []
        while score[a_idx][b_idx] > 0:
            flag = pointer[a_idx][b_idx]
            if flag == '↖':
                inverse_lcs_idxes.append(a_idx - 1)
                if sub_inverse_unmatch_idxes:
                    inverse_unmatch_idxes.append(sub_inverse_unmatch_idxes)
                    sub_inverse_unmatch_idxes = []
                a_idx -= 1
                b_idx -= 1
            elif flag == '←':
                b_idx -= 1
            else:
                sub_inverse_unmatch_idxes.append(a_idx - 1)
                a_idx -= 1
        lcs_idxes = inverse_lcs_idxes[::-1]
        unmatch_idxes = [sium[::-1] for sium in inverse_unmatch_idxes]
        return lcs_idxes, unmatch_idxes
    
    def waterman(self, seq1, seq2):
        m, n = len(seq1), len(seq2)  
        score = self.zeros((m+1, n+1))     
        pointer = self.zeros((m+1, n+1))   
        max_score = 0
        inverse_match_idxes = []
        inverse_unmatch_idxes = []
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                score_diagonal = score[i-1][j-1] + self.match_score(seq1[i-1], seq2[j-1])
                score_up = score[i][j-1] + self.gap_penalty
                score_left = score[i-1][j] + self.gap_penalty
                score[i][j] = max(0,score_left, score_up, score_diagonal)
                if score[i][j] == 0:
                    pointer[i][j] = 0 
                if score[i][j] == score_left:
                    pointer[i][j] = 1 
                if score[i][j] == score_up:
                    pointer[i][j] = 2 
                if score[i][j] == score_diagonal:
                    pointer[i][j] = 3 
                if score[i][j] >= max_score:
                    max_i = i
                    max_j = j
                    max_score = score[i][j];
        
        sub_inverse_unmatch_idxes = []
        i,j = max_i,max_j
        #i, j = m, n    
        while pointer[i][j] != 0:
            if pointer[i][j] == 3:
                inverse_match_idxes.append(i - 1)
                if sub_inverse_unmatch_idxes:
                    inverse_unmatch_idxes.append(sub_inverse_unmatch_idxes)
                    sub_inverse_unmatch_idxes = []
                i -= 1
                j -= 1
            elif pointer[i][j] == 2:
                j -= 1
            elif pointer[i][j] == 1:
                sub_inverse_unmatch_idxes.append(i - 1)
                i -= 1
        max_score = max([max(sc) for sc in score])
        match_idxes = inverse_match_idxes[::-1]
        unmatch_idxes = [sium[::-1] for sium in inverse_unmatch_idxes]
        return match_idxes, unmatch_idxes, max_score
        
class F1:
    def __init__(self):
        pass
        
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

if __name__ == '__main__':	
    a = ['目前', '性能', '比较', '好', '的', '垂直', '起降', '战斗机', '是', '美国', '的', 'F-35', '当中', '的', 'F-35B', '型号']
    b = ['目前', '性能', '比较', '好', '的', '垂直', '起降', '战斗机', '是', '什么', '型号']
    
    #a = 'aacccabbb'
    #b = 'aaa'
    
    #sequence_idxes, unmatch = lcs(a, b)
    #print(''.join([a[t] for t in sequence_idxes]))
    # print(''.join([a[t] for t in unmatch]))
    ali = Alignment()
    ali.waterman(a, b)