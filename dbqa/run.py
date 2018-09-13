# -*- coding:utf8 -*-

import sys
import os
import pickle
import logging
from dataset import BRCDataset
from vocab import DataVocabs
from doc_reader.train import DrqaTrain
from doc_reader.drqa_model import DrqaModel
from utils.prepro import prepro_token, train_test_split
import sys
sys.path.append('../nlp')
from ltpnlp import LtpNlp


class args:
    max_p_num = 1
    max_p_len = 500
    max_q_len = 30
    train_rate = 0.8
    raw_file = './data/cetc/question.json'
    all_file = './data/cetc/all.json'
    train_file = './data/cetc/train.json'
    test_file = './data/cetc/test.json'
    dev_file = './data/cetc/test.json'
    model_dir = './data/cetc/model'
    embed_size = 100
    embedding_dim = 100
    embedding_path = './data/embedding/sogou2012.emb'
    vocab_dir = './data/cetc/'
    batch_size = 32
    log_path = False
    train = True
    cuda = True
    resume = False
    epochs = 20
    eval_per_epoch = 1
    pretrained_words = True
    fix_embeddings = False
    tune_partial = 0
    use_qemb = True
    num_features = 2
    pos = True
    ner = False
    rlr = 0
    optimizer = 'adamax'
    grad_clipping = 10
    weight_decay = 0
    learning_rate = 0.1
    momentum = 0
    tune_partial = 1000
    question_merge = 'self_attn'
    doc_layers = 3
    question_layers = 3
    hidden_size = 128
    concat_rnn_layers = True
    dropout_emb = 0.3
    dropout_rnn = 0.3
    dropout_rnn_output = True
    rnn_type = 'lstm'
    rnn_padding = True
    prepare = True
    log_per_updates = 3
    eval_per_epoch = 1
    max_len = 100
    save_last_only = False
    resume = True
    resume_file = 'best_model.pt'
    predict = True
    resume_options = False
    args_file = './data/cetc/args.pkl'


def preprocess(args):
    ln = LtpNlp()
    tokenizer = ln.tokenize
    prepro_token(args.raw_file, args.all_file, tokenizer, extract_sample=True, chunk='paragraphs')
    train_test_split(args.all_file, args.train_file, args.test_file, args.train_rate)
    ln.release()


def prepare(args):
    logger = logging.getLogger("rc")
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_file, args.dev_file, args.test_file)
    data_vocabs = DataVocabs()
    for word, pos, ner in brc_data.word_iter('train'):
        data_vocabs.word_vocab.add(word)
        data_vocabs.pos_vocab.add(pos)
        data_vocabs.ner_vocab.add(ner)
    unfiltered_vocab_size = data_vocabs.word_vocab.size()
    data_vocabs.word_vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - data_vocabs.word_vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                data_vocabs.word_vocab.size()))
    logger.info('Assigning embeddings...')
    # vocab.randomly_init_embeddings(args.embed_size)
    data_vocabs.word_vocab.load_pretrained_embeddings(args.embedding_path)
    logger.info('embedding size: {}, {}'.format(len(data_vocabs.word_vocab.embeddings), 
                                                len(data_vocabs.word_vocab.embeddings[0])))
    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(data_vocabs, fout)
    logger.info('Done with preparing!')


def train(args):
    logger = logging.getLogger("rc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        data_vocabs = pickle.load(fin)
    args.pos_size = data_vocabs.pos_vocab.size()
    args.ner_size = data_vocabs.ner_vocab.size()
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_file, args.dev_file, args.test_file)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(data_vocabs)
    logger.info('Saving the args')
    pickle.dump(args, open(args.args_file, 'wb'))
    logger.info('Initialize the model...')
    rc_model = DrqaModel(data_vocabs.word_vocab, args)
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
    logger = logging.getLogger("rc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        data_vocabs = pickle.load(fin)
    assert args.test_file, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_file=args.test_file)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(data_vocabs)
    logger.info('Restoring the model...')

    args = pickle.load(open(args.args_file, 'rb'))
    args.pos_size = data_vocabs.pos_vocab.size()
    args.ner_size = data_vocabs.ner_vocab.size()

    rc_model = DrqaModel(data_vocabs.word_vocab,
                         args,
                         eva=True)
    rc_model.evaluate(brc_data)
    # rc_model = RCModel(vocab, args)
    # rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    # logger.info('Predicting answers for test set...')
    # test_batches = brc_data.gen_mini_batches('test', args.batch_size,
    #                                          pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    # rc_model.evaluate(test_batches,
    #                   result_dir=args.result_dir, result_prefix='test.predicted')


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
#
#    if args.prepare:
#        prepare(args)
#        preprocess(args)
#     if args.train:
#         train(args)
    # if args.evaluate:
    #     evaluate(args)
    if args.predict:
        predict(args)


if __name__ == '__main__':
    # prepare(args)
    #batch = train(args)
    run()
