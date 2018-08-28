# -*- coding: utf-8 -*-

from .bleu import Bleu
from .rouge import RougeL

class CetcEva:
    def __init__(self):
        self.rouge_eval = RougeL()
        self.bleu_eval = Bleu()
        
    def eva(self, data, ref_key, cand_key):
        for idx, sample in enumerate(data):
            ref_sent = sample[ref_key]
            cand_sent = sample[cand_key]
            self.rouge_eval.add_inst(cand_sent, ref_sent)
            self.bleu_eval.add_inst(cand_sent, ref_sent)
            print(idx)
        bleu_score = self.bleu_eval.get_score()
        rouge_score = self.rouge_eval.get_score()
        print('bleu score: {}, rouge score: {}'.format(bleu_score, rouge_score))
