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
import msgpack
import pandas as pd
import numpy as np
from .drqa.model import DocReaderModel
from .drqa.utils import str2bool
from .utils import _normalize_answer,_exact_match,_f1_score, score, BatchTransform
from evaluate.cetc_evaluate.main import CetcEva


class DrqaTrain:
    def __init__(self, vocab, args):
        self.args = args
        self.logger = logging.getLogger("rc")
        self.vocab = vocab
        self.batch_transform = BatchTransform(args)
        self.best_val_score = 0
        self.cetc_eva = CetcEva()

    def train(self, data):
        self.logger.info('[training starts.]')
        # train, dev, dev_y, embedding, opt = load_data(vars(args))
        self.logger.info('[train_length:%d dev_length:%d]' % (len(data.train_set),
                                                              len(data.dev_set)))

        embedding = torch.Tensor(list(self.vocab.embeddings))
        if self.args.resume:
            self.logger.info('[loading previous model...]')
            checkpoint = torch.load(os.path.join(self.args.model_dir, self.args.resume))
            if self.args.resume_options:
                opt = checkpoint['config']
            state_dict = checkpoint['state_dict']
            model = DocReaderModel(opt, embedding, state_dict)
            epoch_0 = checkpoint['epoch'] + 1
            for i in range(checkpoint['epoch']):
                random.shuffle(list(range(len(train))))  # synchronize random seed
            if self.args.reduce_lr:
                lr_decay(model.optimizer, lr_decay=self.args.reduce_lr)
        else:
            model = DocReaderModel(self.args, embedding)
        if self.args.cuda:
            model.cuda()
        # if self.args.resume:
        #     batches = BatchGen(dev, opt,batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        #     predictions = []
        #     for batch in batches:
        #         prediction,_ = model.predict(batch)
        #         predictions.extend(prediction)
        #     em, f1 = score(predictions, dev_y)
        #     log.info("[dev EM: {} F1: {}]".format(em, f1))
        #     best_val_score = f1
        # else:
        #     best_val_score = 0.0

        for epoch in range(1, self.args.epochs + 1):
            self.logger.info('Epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            for i, batch in enumerate(train_batches):
                batch = self.batch_transform.transform(batch)
                model.update(batch)
                if i % self.args.log_per_updates == 0:
                    self.logger.info('updates[{0:6}] train loss[{1:.5f}]]'.format(
                        model.updates, model.train_loss.avg))

            if epoch % self.args.eval_per_epoch == 0:
                dev_batches = data.gen_mini_batches('dev', self.args.batch_size, shuffle=True)
                predicts = []
                for batch in dev_batches:
                    transform_batch = self.batch_transform.transform(batch, eva=True)
                    prediction, _ = model.predict(transform_batch)
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
                model.save(model_file, epoch)
                if rouge_score > self.best_val_score:
                    self.best_val_score = rouge_score
                    copyfile(model_file,
                             os.path.join(self.args.model_dir, 'best_model.pt'))
                    self.logger.info('[new best model saved.]')


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer
