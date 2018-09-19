# -*- coding: utf-8 -*-

import json
from doc_reader.drqa_model import DrqaModel
from deep.predict import Predict 
from deep.question2text import Question2text
from run import args
import pickle
import os


if __name__ == '__main__':
    
    data_path = './data/cetc/test.json'
    data_path = './data/cetc/all.json'
    data_path = './data/cetc/all_test.json'
    output_path = './data/deep/test_origin_predict.json'
    args_file = './data/cetc/args.pkl'
    formal_file = './data/deep/all_predict.json'

    model_infos = [('./data/cetc/model/model0', './model0/best_model.pt'),
                   ('./data/cetc/model/model1', './model1/best_model.pt'),
                   ('./data/cetc/model/model2', './model2/best_model.pt'),
                   ('./data/cetc/model/model3', './model3/best_model.pt'), ]

    ensemble_output_adr = './data/cetc/ensemble'

    # args = pickle.load(open(args_file, 'rb'))
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    #     data_vocabs = pickle.load(fin)
    # args.pos_size = data_vocabs.pos_vocab.size()
    # args.ner_size = data_vocabs.ner_vocab.size()

    # rc_model = DrqaModel(data_vocabs.word_vocab,
    #                      args,
    #                      eva=True)
    rc_model = 0
    data_vocabs = 0
    print(data_path)
    pred = Predict(data_path, output_path, data_vocabs, rc_model)
    # pred.predict_formal(formal_file)

    # pred.run(output_path)
    pred.ensemble(model_infos, ensemble_output_adr)
    pred.get_ensemble_result(ensemble_output_adr, output_path)
    pred.predict_formal_from_file(output_path, formal_file)

    #  pred.get_format_json()
    # batch = pred.get_batch(sam)
    # pred.predict(sam)
    # pred.non_find_answer()
