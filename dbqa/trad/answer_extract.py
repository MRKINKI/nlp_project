# -*- coding: utf-8 -*-
from .sim import lcs, waterman

class AnswerExtract:
    def __init__(self, align='lcs'):
        if align == 'lcs':
            self.alignment_model = lcs
        elif align == 'waterman':
            self.alignment_model = waterman
            
    def get_mid_idxes(self, match_idxes, unmatch_idxes):
        mid_idxes = []        
        for sub_unmatch_idxes in unmatch_idxes:
            start, end = sub_unmatch_idxes[0], sub_unmatch_idxes[-1]
            if start-1 in match_idxes and end+1 in match_idxes:
                mid_idxes.extend(sub_unmatch_idxes)
        mid_idxes = sorted(mid_idxes)
        return mid_idxes
    
    def extract(self, text, question):
        match_idxes, unmatch_idxes = self.alignment_model(text, question)
        mid_idxes = self.get_mid_idxes(match_idxes, unmatch_idxes)
        if not unmatch_idxes:
            other_unmatch_idxes = list(range(match_idxes[-1]+1, len(text)))
            if not other_unmatch_idxes:
                other_unmatch_idxes = list(range(0,match_idxes[0]))
            unmatch_idxes.append(other_unmatch_idxes)
        if mid_idxes:
            answer_idxes = mid_idxes
        else:
            answer_idxes = unmatch_idxes[-1]
        return ''.join([text[t] for t in answer_idxes])
