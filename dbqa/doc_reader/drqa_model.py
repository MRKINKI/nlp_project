import re
import os
import sys
import random
import string
import logging
import argparse
from shutil import copyfile
import json
import torch
import json
import msgpack
import pandas as pd
import numpy as np
from .drqa.model import DocReaderModel
from .drqa.utils import str2bool
from .utils import _normalize_answer,_exact_match,_f1_score, score, BatchTransform
from evaluate.cetc_evaluate.main import CetcEva
import pickle


class DrqaModel:
    def __init__(self, vocab, args=None, eva=False):
        self.args = args
        self.logger = logging.getLogger("rc")
        self.best_val_score = 0
        self.cetc_eva = CetcEva()
        embedding = torch.Tensor(list(vocab.embeddings))
        if eva:
            checkpoint = torch.load(os.path.join(self.args.model_dir, self.args.resume_file))
            state_dict = checkpoint['state_dict']
            # self.args = checkpoint['config']
            self.model = DocReaderModel(self.args, embedding, state_dict)
        elif self.args.resume:
            self.logger.info('[loading previous model...]')
            checkpoint = torch.load(os.path.join(self.args.model_dir, self.args.resume_file))
            if self.args.resume_options:
                opt = checkpoint['config']
            state_dict = checkpoint['state_dict']
            self.model = DocReaderModel(opt, embedding, state_dict)
        else:
            self.model = DocReaderModel(self.args, embedding)
        if self.args.cuda:
            self.model.cuda()
        self.batch_transform = BatchTransform(self.args)

    def train(self, data):
        self.logger.info('[training starts.]')
        self.logger.info('[train_length:%d dev_length:%d]' % (len(data.train_set),
                                                              len(data.dev_set)))

        for epoch in range(1, self.args.epochs + 1):
            self.logger.info('Epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            for i, batch in enumerate(train_batches):
                batch = self.batch_transform.transform(batch)
                self.model.update(batch)
                if i % self.args.log_per_updates == 0:
                    self.logger.info('updates[{0:6}] train loss[{1:.5f}]]'.format(
                        self.model.updates, self.model.train_loss.avg))

            if epoch % self.args.eval_per_epoch == 0:
                dev_batches = data.gen_mini_batches('dev', self.args.batch_size, shuffle=True)
                predicts = []
                for batch in dev_batches:
                    transform_batch = self.batch_transform.transform(batch, eva=True)
                    prediction, _ = self.model.predict(transform_batch)
                    for idx, pred in enumerate(prediction):
                        batch[idx]['pred'] = pred
                    predicts.extend(batch)
                bleu_score, rouge_score = self.cetc_eva.eva(predicts, 'answer', 'pred')
                # json.dump(predicts, open('./data/pred.json', 'w'))
                self.logger.info("dev bleu_score: {} rouge_score: {}".format(bleu_score,
                                                                             rouge_score))

            if not self.args.save_last_only:
                model_file = os.path.join(self.args.model_dir,
                                          'checkpoint_epoch_{}_bleu:{}_rouge:{}.pt'.format(epoch,
                                                                                           bleu_score,
                                                                                           rouge_score))
                self.model.save(model_file, epoch)
                if rouge_score > self.best_val_score:
                    self.best_val_score = rouge_score
                    copyfile(model_file,
                             os.path.join(self.args.model_dir, 'best_model.pt'))
                    self.logger.info('[new best model saved.]')

    def evaluate(self, data):
        dev_batches = data.gen_mini_batches('test', self.args.batch_size, shuffle=True)
        predicts = []
        for batch in dev_batches:
            transform_batch = self.batch_transform.transform(batch, eva=True)
            prediction, _ = self.model.predict(transform_batch)
            for idx, pred in enumerate(prediction):
                batch[idx]['pred'] = pred
            predicts.extend(batch)
        bleu_score, rouge_score = self.cetc_eva.eva(predicts, 'answer', 'pred')
        # json.dump(predicts, open('./data/pred.json', 'w'))
        self.logger.info("dev bleu_score: {} rouge_score: {}".format(bleu_score,
                                                                     rouge_score))

    def predict(self, batch):
        transform_batch = self.batch_transform.transform(batch, eva=True)
        predictions, scores = self.model.predict(transform_batch)
        return predictions, scores
