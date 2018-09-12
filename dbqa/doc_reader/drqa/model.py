# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import json

from torch.autograd import Variable
from .utils import AverageMeter
from .rnn_reader import RnnDocReader

logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """
    qa主模型。包含神经网络结构构建、预测及训练部分。
    """

    def __init__(self, opt, embedding=None, state_dict=None):

        self.opt = opt.__dict__
        opt = opt.__dict__
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()

        #神经网络结构
        self.network = RnnDocReader(opt, embedding=embedding)
        
        #加载模型参数，训练时使用
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        #优化器
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def update(self, ex):
        #训练模式
        self.network.train()

        #是否使用GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:7]]
            target_s = Variable(ex[7].cuda(async=True))
            target_e = Variable(ex[8].cuda(async=True))
        else:
            inputs = [Variable(e) for e in ex[:7]]
            target_s = Variable(ex[7])
            target_e = Variable(ex[8])

        #网络输出
        score_s, score_e = self.network(*inputs)

        #交叉熵损失计算
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.data[0], ex[0].size(0))

        #反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients 防止梯度爆炸
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.opt['grad_clipping'])

        #更新参数
        self.optimizer.step()
        self.updates += 1

        #重置不用finetue的词对应的词向量为原始词向量
        self.reset_parameters()

    def predict(self, ex):
        #评估模式
        self.network.eval()

        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:7]]
        else:
            inputs = [Variable(e, volatile=True) for e in ex[:7]]


        score_s, score_e = self.network(*inputs)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        #获得文档分过词的文本数据
        text = ex[-1]
        
        #预测文档中词作为答案初始及结束的概率
        predictions = []
        max_scores = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            max_scores.append(np.max(scores))
            predictions.append(''.join(text[i][s_idx:e_idx+1]))

        return predictions,max_scores

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.get_args_dict(self.opt),
            'epoch': epoch
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    @staticmethod
    def get_args_dict(obj):
        pr = {}
        for name in obj:
            value = obj[name]
            if not name.startswith('__') and not callable(value):
                pr[name] = value
        return pr

    def cuda(self):
        self.network.cuda()
