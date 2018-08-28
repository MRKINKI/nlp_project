# -*- coding: utf-8 -*-
import json


data_path = '../data/trad/all.json'
prepro_samples = []
with open(data_path, encoding='utf-8') as fin:
    for idx, line in enumerate(fin):
        sample = json.loads(line.strip())
        prepro_samples.append(sample)
        if idx > 1000:
            break





