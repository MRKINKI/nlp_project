# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
from utils.feature import FeatureExtract


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_file, dev_file, test_file):
        self.feature_extract = FeatureExtract()
        
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_file:
            self.train_set = self._load_dataset(train_file, train=True)

        if dev_file:
            self.dev_set = self._load_dataset(dev_file)

#        if test_files:
#            for test_file in test_files:
#                self.test_set += self._load_dataset(test_file)

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path,encoding = 'utf8') as fin:
            data_set = []
            for idx, line in enumerate(fin):
                sample = json.loads(line.strip())
                for qa_sample in sample['questions']:
                    if qa_sample['bad_sample'] == 0 and qa_sample['find_answer'] == 1:
                        # print(idx, qa_sample.keys(), qa_sample['find_answer'])
                        new_qa_sample = self.feature_extract.extract(qa_sample)
                        data_set.append(new_qa_sample)
        return data_set

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        raw_data = [data[i] for i in indices]
        batches = []
        for sidx, sample in enumerate(raw_data):
            batch_data = {}
            batch_data['question_word_ids'] = sample['question_word_ids']
            batch_data['context_word_ids'] = sample['context_word_ids']
#            if 'answer_spans' in sample:
#                batch_data['start_id'] = sample['answer_spans'][0]
#                batch_data['end_id'] = sample['answer_spans'][1]
#            else:
#                batch_data['start_id'] = 0
#                batch_data['end_id'] = 0
            batches.append(batch_data)
        return batches

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_word']:
                    yield token
                for token in sample['context_word']:
                    yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_word_ids'] = vocab.convert_to_ids(sample['question_word'])
                sample['context_word_ids'] = vocab.convert_to_ids(sample['context_word'])
#                for passage in sample['passages']:
#                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)
