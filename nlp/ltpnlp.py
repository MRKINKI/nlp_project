# -*- coding: utf-8 -*-
import os
from pyltp import Segmentor,Postagger,NamedEntityRecognizer,Parser
from char_map import CharMap


class LtpNlp:
    def __init__(self, model_dir='d:/MyProject/data/ltp_data_v3.4.0'):
        self.model2infos = {'cws': ['cws.model', Segmentor], 
                            'pos': ['pos.model', Postagger],
                            'ner': ['ner.model', NamedEntityRecognizer],
                            'parser': ['parser.model', Parser]}
        self.model_dir = model_dir
        self.model = self.load_model()
        self.model_pipeline = [('cws', self.model['cws'].segment, ['text']), 
                               ('pos', self.model['pos'].postag, ['cws']), 
                               ('ner', self.model['ner'].recognize, ['cws', 'pos']),
                               ('parser', self.model['parser'].parse, ['cws', 'pos'])]
        self.char_map = CharMap()
        #print(dir(self.model['cws']))

    def load_model(self):
        model_dict = {}
        for name, model_info in self.model2infos.items():
            model = model_info[1]()
            model.load(os.path.join(self.model_dir, model_info[0]))
            model_dict[name] = model
        return model_dict
        
    def release(self):
        for _, model in self.model.items():
            model.release()
    
    def tokenize(self, text, method='cws'):
        sequence = dict()
        text = self.char_map.multiple_replace(text)
        sequence['text'] = text
        for m, method_fun, args in self.model_pipeline:
            sequence[m] = list(method_fun(*[sequence[arg] for arg in args]))
            if m == method:
                break
        return sequence
        
    
if __name__ == '__main__':
    ln = LtpNlp()
    sentence = '姚明出生在中国上海'
    tokens = ln.tokenize(sentence, 'pos')
    print(tokens)
    ln.release()
