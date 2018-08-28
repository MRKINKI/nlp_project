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
import numpy as np
from drqa1.model import DocReaderModel
from drqa1.utils import str2bool
from utils import _normalize_answer,_exact_match,_f1_score,score,BatchGen

parser = argparse.ArgumentParser(
    description='start training'
)

parser.add_argument('--log_file', default='output.log',
                    help='log文件地址')
parser.add_argument('--log_per_updates', type=int, default=3,
                    help='输出训练误差的间隔batch数')
parser.add_argument('--data_file', default='data/data.msgpack',
                    help='数据地址')
parser.add_argument('--model_dir', default='models',
                    help='模型地址')
parser.add_argument('--save_last_only', action='store_true',
                    help='是否只存储最终模型')
parser.add_argument('--eval_per_epoch', type=int, default=1,
                    help='输出验证误差间隔epoch数')
parser.add_argument('--seed', type=int, default=411,
                    help='随机种子值')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='默认使用GPU')
parser.add_argument('-e', '--epochs', type=int, default=20)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-rs', '--resume', default='',
                    help='previous model file name (in `model_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
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
                    help='finetune词频排前tp的词,默认1000')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
parser.add_argument('--rnn_padding', action='store_true',
                    help='perform rnn padding (much slower but more accurate).')
# model
parser.add_argument('--question_merge', default='self_attn')
parser.add_argument('--doc_layers', type=int, default=3)
parser.add_argument('--question_layers', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_features', type=int, default=2)

parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=False,
                    help='是否使用词性')
parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=False,
                    help='是否使用实体标注')
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
parser.add_argument('--pretrained_words', type=str2bool,default = True)
parser.add_argument('--random_embedding', type=str2bool,default = True,
                    help='是否使用随机矩阵作为词embedding矩阵')

args = parser.parse_args()

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)


def main():
    log.info('[program starts.]')
    log.info(vars(args))
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    
    log.info('[Data loaded.]')
    log.info('[train_length:%d dev_length:%d]'%(len(train),len(dev)))

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(model_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            random.shuffle(list(range(len(train))))  # synchronize random seed
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
            
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1

    if args.cuda:
        model.cuda()

    if args.resume:
        batches = BatchGen(dev, opt,batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        predictions = []
        for batch in batches:
            prediction,_ = model.predict(batch)
            predictions.extend(prediction)
        em, f1 = score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        best_val_score = f1
    else:
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warn('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train, opt,batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
                
         #eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen(dev, opt,batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
            predictions = []
            for batch in batches:
                prediction,_ = model.predict(batch)
                predictions.extend(prediction)
            em, f1 = score(predictions, dev_y)
            log.info("dev EM: {} F1: {}".format(em, f1))
            
         #save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{}_EM:{}_F1:{}.pt'.format(epoch,em,f1))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer


def load_data(opt):
    with open('data/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = meta['embedding']
    opt['pos_size'] = len(meta['vocab_tag'])
    opt['ner_size'] = len(meta['vocab_ent'])
    #embedding_raw = [[float(t1) for t1 in t] for t in embedding_raw]

    embedding = torch.Tensor(embedding)
    if opt['random_embedding']:
        embedding = list(np.random.randn(embedding.size(0),embedding.size(1)))
        embedding = torch.Tensor(embedding)
    
#    embedding = torch.Tensor(meta['embedding'])
#    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    
    log.info('vocab_size:{}  embedding_dim:{}'.format(opt['vocab_size'],opt['embedding_dim']))
    
    if not opt['fix_embeddings']:
#        embedding[1] = torch.zeros(opt['embedding_dim'])
        embedding[1] = torch.normal(mean=torch.zeros(opt['embedding_dim']), std=1.0)
    with open(args.data_file, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    train_orig = pd.read_csv('data/train.csv')
    dev_orig = pd.read_csv('data/dev.csv')
    train = list(zip(
        data['trn_context_ids'],
        data['trn_context_features'],
        data['trn_context_tags'],
        data['trn_context_ents'],
        data['trn_question_ids'],
        train_orig['answer_start_token'].tolist(),
        train_orig['answer_end_token'].tolist(),
        data['trn_context_text'],
    ))
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_context_features'],
        data['dev_context_tags'],
        data['dev_context_ents'],
        data['dev_question_ids'],
        data['dev_context_text'],
    ))
    dev_y = dev_orig['answers'].tolist()[:len(dev)]
    dev_y = [eval(y) for y in dev_y]
    return train, dev, dev_y, embedding, opt


if __name__ == '__main__':
    main()

