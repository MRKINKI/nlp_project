# -*- coding: utf-8 -*-

import json
from doc_reader.drqa_model import DrqaModel
from deep.predict import Predict 
from deep.question2text import Question2text
from run import args
import pickle
import os


if __name__ == '__main__':
    with open('./data/cetc/test.json', encoding='utf-8') as fin:
        for line in fin:
            sam = json.loads(line.strip())
            break
    # qt = Question2text()
    # mrc = qt.find_best_question_match(sam['paragraphs_tokens'], sam['questions'][0]['question_tokens'])
    
    data_path = './data/cetc/test.json'
    # data_path = './data/cetc/all.json'
    output_path = './data/deep/origin_predict.json'
    args_file = './data/cetc/args.pkl'
    args = pickle.load(open(args_file, 'rb'))
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        data_vocabs = pickle.load(fin)
    args.pos_size = data_vocabs.pos_vocab.size()
    args.ner_size = data_vocabs.ner_vocab.size()

    rc_model = DrqaModel(data_vocabs.word_vocab,
                         args,
                         eva=True)
    pred = Predict(data_path, output_path, data_vocabs, rc_model)
    # pred.predict_formal()
    pred.run()
    pred.get_format_json()
    # batch = pred.get_batch(sam)
    # pred.predict(sam)
    # pred.non_find_answer()
