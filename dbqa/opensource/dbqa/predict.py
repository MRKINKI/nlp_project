# -*- coding: utf-8 -*-

import re
import os
import sys
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pandas as pd
from drqa1.model import DocReaderModel
from drqa1.utils import str2bool
from utils import _normalize_answer,_exact_match,_f1_score,score,BatchGen
from utils import load_origin_data,load_data_tokenize,token2id,get_context_features
import pymongo
from tokenizers.ltp_tokenizer import LtpTokenizer
from tokenizers.jieba_origin_tokenizer import JiebaOriginTokenizer
#from tokenizers.jieba_tokenizer import JiebaTokenizer

parser = argparse.ArgumentParser(
    description='use Document Reader model to predict.'
)

parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--model_dir', default='models',
                    help='path to store saved models.')
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
parser.add_argument('--rnn_padding', action='store_true',
                    help='perform rnn padding (much slower but more accurate).')
parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                    help='reduce initial (resumed) learning rate by this factor.')
parser.add_argument('-op', '--optimizer', default='adamax',
                    help='supported optimizer: adamax, sgd')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                    help='only applied to SGD.')
parser.add_argument('-mm', '--momentum', type=float, default=0,
                    help='only applied to SGD.')
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings.')
parser.add_argument('--question_merge', default='self_attn')
parser.add_argument('--doc_layers', type=int, default=3)
parser.add_argument('--question_layers', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=False,
                    help='use pos tags as a feature.')
parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=False,
                    help='use named entity tags as a feature.')
parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--dropout_emb', type=float, default=0.3)
parser.add_argument('--dropout_rnn', type=float, default=0.3)
parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--max_len', type=int, default=15)
parser.add_argument('--rnn_type', default='lstm',
                    help='supported types: rnn, gru, lstm')
parser.add_argument('-rs', '--resume', default='best_model.pt',
                    help='previous model file name (in `model_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('--embedding_dim', type=int, default=100,
                    help='max question length')
parser.add_argument('--raw_data', type=str2bool, default=True,
                    help = '采取webQA原始词向量及分词')
parser.add_argument('--max_doc_len', type=int, default=500,
                    help='文档最大长度')
parser.add_argument('--max_que_len', type=int, default=30,
                    help='问题最大长度')
parser.add_argument('--tokenizer',default = 'ltp',
                    help = '分词工具 可选：ltp、jieba')
parser.add_argument('--replace', type=str2bool, default=False)
parser.add_argument('--test_collection_source', default='WebQa_test')

args = parser.parse_args()

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set mongo
client = pymongo.MongoClient(host = '192.168.1.145', port = 27017)
db = client['AttnReader']
collection_test = db[args.test_collection_source]

# setup logger
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def main():
    log.info('[program starts.]')
    log.info(vars(args))
    test, test_y, embedding, opt,test_data = load_data(vars(args))
    
    log.info('[Data loaded.]')
    log.info('[loading previous model...]')
    
    checkpoint = torch.load(os.path.join(model_dir, args.resume))
    state_dict = checkpoint['state_dict']
    model = DocReaderModel(opt, embedding, state_dict)
            
    if args.cuda:
        model.cuda()

    batches = BatchGen(test, opt,batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    predictions = []
    max_scores = []
    for batch in batches:
        prediction,max_score = model.predict(batch)
        predictions.extend(prediction)
        max_scores.extend(max_score)
    em, f1 = score(predictions, test_y)
    log.info("[test EM: {} F1: {}]".format(em, f1))
    
    test_data['question'] = test_data['query_words'].apply(lambda x:''.join(x))
    test_data['document'] = test_data['words'].apply(lambda x:''.join(x))
    test_data['answer'] = test_data['answers'].apply(lambda x:x[0])
    
    result = test_data[['question','document','answer']]
    result['predict'] = predictions
    result['pro'] = max_scores
    result.to_csv('test_result.csv',index = False)
    

def load_data(opt):
    if args.raw_data:
        test = load_origin_data(collection_test,args,single_answer = False)
    else:
        if args.tokenizer == 'ltp':
            tokenizer = LtpTokenizer('ltp_data')
        if args.tokenizer == 'jieba':
            tokenizer =JiebaTokenizer()
        if args.tokenizer == 'jieba_origin':
            tokenizer = JiebaOriginTokenizer()
        test = load_data_tokenize(collection_test,tokenizer,args,single_answer = False)
    
    log.info("[test data length:{}]".format(len(test)))
    
    with open('data/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    vocab = meta['vocab']
    vocab_tag = meta['vocab_tag']
    vocab_ent = meta['vocab_ent']

    embedding = meta['embedding']
    embedding = torch.Tensor(embedding)
    
    question_tokens = list(test.query_words)
    context_tokens = list(test.words)
    context_tags = list(test.postags)
    context_ents = list(test.netags)

    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    embedding[1] = torch.normal(means=torch.zeros(opt['embedding_dim']), std=1.)

    question_ids = token2id(question_tokens, vocab, unk_id=1)
    context_ids = token2id(context_tokens, vocab, unk_id=1)
    context_features = get_context_features(question_tokens,context_tokens)
    
    context_tag_ids = token2id(context_tags, vocab_tag)
    context_ent_ids = token2id(context_ents, vocab_ent)
    
    test_batches = list(zip(
        context_ids,
        context_features,
        context_tag_ids,
        context_ent_ids,
        question_ids,
        context_tokens,
    ))
    
    test_y = test['answers'].tolist()[:len(test)]
    return test_batches, test_y, embedding, opt,test

 
if __name__ == '__main__':
    main()
    
    
    