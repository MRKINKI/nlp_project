# -*- coding:utf8 -*-

import sys
import os
import pickle
import logging
from dataset import BRCDataset
from vocab import Vocab
from doc_reader.train import DrqaTrain


class args:
    max_p_num = 1
    max_p_len = 500
    max_q_len = 30
    train_files = ['./data/cetc/train.json']
    test_files = ['./data/cetc/test.json']
    dev_files = ['./data/cetc/test.json']
    embed_size = 100
    embedding_path = './data/embedding/sogou2012.emb'
    vocab_dir = './data/cetc/'
    batch_size = 32
    log_path = False
    train = True
    cuda = True
    resume = False
    epochs = 20
    eval_per_epoch = 1
    resume = False
    pretrained_words = True
    fix_embeddings = True


def prepare(args):
    logger = logging.getLogger("rc")
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)
    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                vocab.size()))
    logger.info('Assigning embeddings...')
    # vocab.randomly_init_embeddings(args.embed_size)
    vocab.load_pretrained_embeddings(args.embedding_path)
    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)
    logger.info('Done with preparing!')


def train(args):
    logger = logging.getLogger("rc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    train_batches = brc_data.gen_mini_batches('train', args.batch_size, shuffle=True)
    for batch in train_batches:
        # print(batch)
        break
    rc_model = DrqaTrain(vocab, args)
    logger.info('Training the model...')
    rc_model.train(brc_data)
#    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
#                   save_prefix=args.algo,
#                   dropout_keep_prob=args.dropout_keep_prob)
#    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    # args = parse_args()

    logger = logging.getLogger("rc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

#    args.prepare = True

    # if args.prepare:
    #     prepare(args)
    if args.train:
        train(args)
    # if args.evaluate:
    #     evaluate(args)
    # if args.predict:
    #     predict(args)

if __name__ == '__main__':
    # prepare(args)
    # batch = train(args)
    run()
