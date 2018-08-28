# -*- coding: utf-8 -*-

from evaluate.cetc_evaluate.main import CetcEva
import json

if __name__ == '__main__':
    cetc_eva = CetcEva()
    data = json.load(open('./data/trad/local_all_predict.json'))
    cetc_eva.eva(data, 'true_answer', 'extract_answer')