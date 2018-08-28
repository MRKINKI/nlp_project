# -*- coding: utf-8 -*-
import re
import json
import os
import msgpack
import numpy as np
import pandas as pd
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from drqa1.utils import str2bool
import logging
import pymongo
import tokenizers
from utils import documentfilter,normalize_text,token2id,load_origin_data
from utils import load_data_tokenize,get_context_features,build_embedding

parser = argparse.ArgumentParser(
    description='Preprocessing data files'
)
parser.add_argument('--sort_all', action='store_true',
                    help='高频词选择考虑问题及文档')
parser.add_argument('--sample_size', type=int, default=0,
                    help='采样比例.')
parser.add_argument('--max_doc_len', type=int, default=500,
                    help='文档最大长度')
parser.add_argument('--max_que_len', type=int, default=30,
                    help='问题最大长度')
parser.add_argument('--embedding_dim', type=int, default=64,
                    help='词向量维度')
parser.add_argument('--raw_data', type=str2bool, default=False,
                    help = '采取webQA原始词向量及分词')
parser.add_argument('--replace', type=str2bool, default=False)
parser.add_argument('--wv_file',default = 'wordvecs_64_replace.txt',
                    help = '词向量位置')
parser.add_argument('--tokenizer',default = 'ltp',
                    help = '分词工具 可选：ltp、jieba_origin、character')
parser.add_argument('--single_answer_train',type=str2bool,default=True,
                    help = '训练时候是否单一答案')
parser.add_argument('--collection_train_name',default='WebQa_train',
                    help = '')
parser.add_argument('--collection_val_name',default='WebQa_val',
                    help = '')
args = parser.parse_args()


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)
log.info(vars(args))
log.info('start data preparing...')


def load_wv_vocab(vocab_file):
    '''
    从词向量文件中加载词典
    '''
    vocab = set()
    with open(vocab_file,encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-args.embedding_dim]))  
            vocab.add(token)
    return vocab

def build_vocab(questions, contexts, wv_vocab):
    '''
    统计问题及文档中的词频并排序，可以采取两种情况方式：
    1、文档和问题集合中的词平等统计
    2、优先排列问题中的词，之后再排列文档中的词
    questions:
        问题词list
    contexts:
        问档词list
    wv_vocab:
        词向量词典
    '''
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
        
#        vocab_old = vocab
#        vocab += [t for t in wv_vocab if t not in vocab_old]
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter


def get_vocab(wlist):
    '''
    统计实体标记及词性的类别
    '''
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)
    feature_dict = {}
    for ex in wlist:
        for w in ex:
            _insert(w)
    return feature_dict    


if __name__ == '__main__':

    client = pymongo.MongoClient(host='192.168.1.145', port=27017)
    db = client['AttnReader']
    collection_train = db[args.collection_train_name]
    collection_val = db[args.collection_val_name]

    if args.raw_data:
        train = load_origin_data(collection_train,args,single_answer = args.single_answer_train)
        dev = load_origin_data(collection_val,args,single_answer = False)
    else:
        tokenizer = tokenizers.get_class(args.tokenizer)()
        train = load_data_tokenize(collection_train,tokenizer,args,single_answer = args.single_answer_train)
        dev = load_data_tokenize(collection_val,tokenizer,args,single_answer = False)
        if args.tokenizer == 'ltp':
            tokenizer.release()
    
    context_ents = list(train.netags) + list(dev.netags)
    context_tags = list(train.postags) + list(dev.postags)
    
    question_tokens = list(train.query_words) + list(dev.query_words)
    context_tokens = list(train.words) + list(dev.words)
    
    #加载词向量文件中词典
    wv_vocab = load_wv_vocab(os.path.join('embedding',args.wv_file))
    log.info('wv_vocab loaded.vocab_size：'.format(len(wv_vocab)))
    
    #建立训练、验证数据集对应词表
    vocab, counter = build_vocab(question_tokens, context_tokens,wv_vocab)
    
    #加载词向量文件中词向量
    embedding = build_embedding(os.path.join('embedding',args.wv_file), vocab, args.embedding_dim)
    log.info('got embedding matrix.')
    
    #建立词到序号的映射
    question_ids = token2id(question_tokens, vocab, unk_id=1)
    context_ids = token2id(context_tokens, vocab, unk_id=1)
    
    #提取文档中词的特征
    context_features = get_context_features(question_tokens,context_tokens)
    
    #数据集中词性及实体标记的类别
    vocab_tag = get_vocab(context_tags)
    vocab_ent = get_vocab(context_ents)
    
    log.info('Found {} POS tags: {}'.format(len(vocab_tag),vocab_tag))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

    #建立词性、实体标记到序号的映射
    context_tag_ids = token2id(context_tags, vocab_tag)
    context_ent_ids = token2id(context_ents, vocab_ent)
    
    #处理过后的数据存储
    train.to_csv('data/train.csv', index=False)
    dev.to_csv('data/dev.csv', index=False)
    
    log.info('train size:{} dev size:{}'.format(len(train), len(dev)))
    
    meta = {
        'vocab': vocab,
        'vocab_tag':vocab_tag,
        'vocab_ent':vocab_ent,
        'embedding': embedding.tolist()
    }
    
    with open('data/meta.msgpack', 'wb') as f:
        msgpack.dump(meta, f)
    
    result = {
        'trn_question_ids': question_ids[:len(train)],
        'dev_question_ids': question_ids[len(train):],
        'trn_context_ids': context_ids[:len(train)],
        'dev_context_ids': context_ids[len(train):],
        'trn_context_features': context_features[:len(train)],
        'dev_context_features': context_features[len(train):],
        'trn_context_text': context_tokens[:len(train)],
        'dev_context_text': context_tokens[len(train):],
        'trn_context_tags': context_tag_ids[:len(train)],
        'dev_context_tags': context_tag_ids[len(train):],
        'trn_context_ents': context_ent_ids[:len(train)],
        'dev_context_ents': context_ent_ids[len(train):]
    }

    with open('data/data.msgpack', 'wb') as f:
        msgpack.dump(result, f)
        
    log.info('saved to disk.')


    
