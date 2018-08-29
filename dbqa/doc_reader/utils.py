# -*- coding: utf-8 -*-

import re
import json
import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from .drqa.utils import str2bool
import logging
import pymongo
from collections import Counter
import string
from gensim.models.word2vec import Word2Vec
import torch
import random
import unicodedata


def build_embedding(embed_file, targ_vocab, dim_vec):
    '''
    构建embedding矩阵，矩阵的列序号对应词的索引。
    embed_file:
        词向量文件
    targ_vocab:
        词典
    dim_vec:
        词向量维度
    '''
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, dim_vec))
    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file,encoding='utf8') as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-dim_vec]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-dim_vec:]]
    return emb


def documentfilter(ds,args,single_answer):
    '''
    样本过滤 考虑长度及答案在文档中出现次数
    '''
    g = []
    for wd in ds:
        if len(wd['words']) < args.max_doc_len and len(wd['query_words']) < args.max_que_len and len(wd['words']) >0:
            if single_answer:
                if len(wd['answers_index'])==1:
                    g.append(wd)
            elif len(wd['answers_index'])>=1:
                g.append(wd)
    return g    

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_origin_data(collection,args,single_answer):
    '''
    从数据库中加载数据，并标记答案在文档中的索引
    '''
    data = np.array(documentfilter(collection.find(),args,single_answer))
    #data = np.array(documentfilter(collection.find().limit(3000),args,single_answer))
    data = pd.DataFrame(list(data))
    if single_answer:
        data['answer_start_token'] = data['answers_index'].apply(lambda x:x[0][0])
        data['answer_end_token'] = data['answers_index'].apply(lambda x:x[0][1])
    else:
        data['answer_start_token'] = data['answers_index'].apply(lambda x:[t[0] for t in x])
        data['answer_end_token'] = data['answers_index'].apply(lambda x:[t[1] for t in x])
    answers = []
    if single_answer:
        for i in range(len(data)):
            start = data['answer_start_token'][i]
            end = data['answer_end_token'][i]
            answer = data['words'][i][start:end+1]
            answers.append([''.join(answer)])
    else:
        for i in range(len(data)):
            start = data['answer_start_token'][i]
            end = data['answer_end_token'][i]
            for s,e in zip(start,end):
                answer = data['words'][i][s:e+1]
            answers.append([''.join(answer)])
    data['answers'] = answers         
    return data
    
def token2id(docs, vocab, unk_id=None):
    '''
    将词序列转换成索引序列，未登录词转为unk_id
    '''
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids


def get_context_features(question_tokens,context_tokens):
    '''
    提取问答词特征(词频及是否出现在问题中)
    '''
    context_features = []
    context_tf = []
    for doc in context_tokens:
        counter_ = collections.Counter(w.lower() for w in doc)
        total = sum(counter_.values())
        context_tf.append([counter_[w.lower()] / total for w in doc])
    
    #is in question
    for question,context in zip(question_tokens,context_tokens):
        match_origin = [t in question for t in context]
        context_features.append(list(zip(match_origin)))
    
    context_features = [[list(w) + [tf] for w, tf in zip(doc, tfs)] for doc, tfs in
                        zip(context_features, context_tf)]
    
    return context_features
    
    
def load_data_tokenize(collection,tokenizer,args,single_answer):
    '''
    调用分词工具重新分词并提取实体标记及词性特征(如果按照webqa数据集原始分词则不需
    调用此函数)
    '''
    data = load_origin_data(collection,args,single_answer)
    
    def get_position(s1,s2):
        m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  
        mmax=0   
        p=0  
        for i in range(len(s1)):  
            for j in range(len(s2)):  
                if s1[i]==s2[j]:  
                    m[i+1][j+1]=m[i][j]+1  
                    if m[i+1][j+1]>mmax:  
                        mmax=m[i+1][j+1]  
                        p=i+1
        return p-mmax,p-1,mmax
                
    #tokenizer = LtpTokenizer('ltp_data')
    documents = list(data['words'].apply(lambda x:''.join(x)))
    answers = list(data['answers'].apply(lambda x:x[0]))
    questions = list(data['query_words'].apply(lambda x:''.join(x)))
    
    netags = []
    postags = []
    document_words = []
    answer_words = []
    question_words = []
    new_answer_word = []    
    answer_start_token = []
    answer_end_token = []
    
    for doc in documents:
        token = tokenizer.tokenize(doc)
        postag = [t[1] for t in token]
        netag = [t[2] for t in token]
        document_word = []
        postags.append(postag)
        netags.append(netag)
        for p,w in zip(postag,[t[0] for t in token]):
            if args.replace:
                if p == 'nt':
                    w = '<nt>'
                if p == 'm':
                    w = '<m>'
            document_word.append(w)
        document_words.append(document_word)

    for que in questions:
        token = tokenizer.tokenize(que)
        question_words.append([t[0] for t in token])
        
    for ans in answers:
        token = tokenizer.tokenize(ans)
        answer_words.append([t[0] for t in token])
        
    for raw_answer,answer,word in zip(answers,answer_words,document_words):
        if raw_answer in word:
            new_answer_word.append(raw_answer)
            answer_start_token.append(word.index(raw_answer))
            answer_end_token.append(word.index(raw_answer))
        else:
            if len(answer) == 1:
                k = 1
                answer = answer[0]
                for w in word:
                    if answer in w:
                        k = 0
                        new_answer_word.append(w)
                        answer_start_token.append(word.index(w))
                        answer_end_token.append(word.index(w))
                        break
                if k == 1:
                    new_answer_word.append(-2)
                    answer_start_token.append(-2)
                    answer_end_token.append(-2)
            else:
                st,et,m = get_position(word,answer)
                if m == len(answer):
                    new_answer_word.append(''.join(answer))
                    answer_start_token.append(st)
                    answer_end_token.append(et)
                else:
                    new_answer_word.append(-1)
                    answer_start_token.append(-1)
                    answer_end_token.append(-1)

    answers = [[t] for t in new_answer_word]
    (data['words'],data['new_answers'],data['answer_start_token'],
    data['answer_end_token'],data['netags'],data['postags'],
    data['query_words'],data['answers'] ) = (document_words,
                                            new_answer_word,
                                            answer_start_token,
                                            answer_end_token,
                                            netags,
                                            postags,
                                            question_words,
                                            answers)
    
    print('-1：%f'%(len(data[data['new_answers'] == -1])/len(data)))
    print('-2：%f'%(len(data[data['new_answers'] == -2])/len(data)))
    
    data = data[(data['new_answers'] != -2)&(data['new_answers'] != -1)]
    return data


class BatchGen:
    def __init__(self, data,opt ,batch_size, gpu, evaluation=False):
        '''
        input:
            data - list of lists
            batch_size - int
        '''
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu
        self.opt = opt
        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 6
            else:
                assert len(batch) == 8

            context_len = max(len(x) for x in batch[0])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[1][0][0])
            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.opt['pos_size']).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1
                        
            context_ent = torch.Tensor(batch_size, context_len, self.opt['ner_size']).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1
#                context_ent[i, :len(doc)] = torch.LongTensor(doc)
                
            question_len = max(len(x) for x in batch[4])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[4]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            if not self.eval:
                y_s = torch.LongTensor([int(t) for t in batch[5]])
                y_e = torch.LongTensor([int(t) for t in batch[6]])
            text = list(batch[-1])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text)

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    '''
    计算EM
    '''
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False

def _f1_score(pred, answers):
    '''
    计算F1
    '''
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    #g_tokens = _normalize_answer(pred).split()
    g_tokens = [t for t in _normalize_answer(pred)]
    #scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    scores = [_score(g_tokens, [t for t in _normalize_answer(a)]) for a in answers]
    return max(scores)

def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1   
