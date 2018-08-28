# -*- coding: utf-8 -*-
import json
import collections
import torch
from trad.sim import waterman

file = './data/cetc/question.json'

#data = json.load(open(file, encoding='utf-8'))
#qtype_dict = {}
#question_dict = collections.defaultdict(list)
#samples = data[:100]
#
#sample = samples[0]
#
#article_content = sample['article_content']
#
#answer = '第一，是高度的隐身第二， 雷电之神的个头比较大'
#
#match_idxes, unmatch_idxes = waterman(article_content, answer)
#
#match_sequence = ''.join([article_content[i] for i in match_idxes])

#for sample in data:
#    questions = sample['questions']
#    for qdict in questions:
#        qtype = qdict['question_type']
#        q = qdict['question']
#        qtype_dict[qtype] = qtype_dict.get(qtype, 0) + 1
#        question_dict[qtype].append(q)
        

data_path = './data/cetc/data.json'
prepro_samples = []
with open(data_path, encoding='utf-8') as fin:
    for idx, line in enumerate(fin):
        sample = json.loads(line.strip())
        prepro_samples.append(sample)
        if idx > 100:
            break

#newsamples = get_sample(sample)


