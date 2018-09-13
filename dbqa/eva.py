# -*- coding: utf-8 -*-

from evaluate.cetc_evaluate.main import CetcEva
import json

if __name__ == '__main__':
    path = './data/trad/local_all_predict.json'
    path = './data/deep/format_predict.json'
    cetc_eva = CetcEva()
    data = json.load(open(path, encoding='utf-8'))
    bleu_score, rouge_score = cetc_eva.eva(data, 'true_answer', 'extract_answer')
    print('bleu score: {}, rouge score: {}'.format(bleu_score, rouge_score))
