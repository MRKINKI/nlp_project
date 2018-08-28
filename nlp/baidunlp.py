# -*- coding: utf-8 -*-

import json
import requests


class BaiduNlp:
    def __init__(self):
        self.ak = '59WUD5pn2OIUTmpD5hXGGTQB'
        self.sk = 'TV8qkfss2D5G0XnYsolSnY5VqMx5LE9a'
        self.access_token_host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'
#        self.headers = {'Content-Type': 'application/json; charset=UTF-8'}
        self.headers = {'Content-Type': 'application/json'} 
        self.lex_host = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/lexer'
        self.dep_host = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/depparser'
        self.emb_host = 'https://aip.baidubce.com/rpc/2.0/nlp/v2/word_emb_vec'
        self.access_token = self.get_access_token()['access_token']
        
    def get_access_token(self):
        host = self.access_token_host.format(self.ak, self.sk)
        return json.loads(requests.post(host, headers=self.headers).text)
        
    def lexer(self, text):
        post_data = bytes(json.dumps({'text': text}), encoding='gbk')
        url = self.lex_host+'?access_token='+self.access_token
        return json.loads(requests.post(url, data=post_data, headers=self.headers).text)
        
    def depparser(self, text):
        post_data = bytes(json.dumps({'text': text, 'mode': 0}), encoding='gbk')
        url = self.dep_host+'?access_token='+self.access_token
        return json.loads(requests.post(url, data=post_data, headers=self.headers).text)
        
    def word_emb(self, text):
        post_data = bytes(json.dumps({'word': text}), encoding='gbk')
        url = self.emb_host+'?access_token='+self.access_token
        return json.loads(requests.post(url, data=post_data, headers=self.headers).text)


if __name__ == '__main__':
    bn = BaiduNlp()
    word = "毛泽东"
    sentence = "刘建国-上海奔腾企业(集团)有限公司董事长介绍→买购网"
    aaa = bn.get_access_token()
    bbb = bn.lexer(sentence)
    cc = bn.depparser(sentence)
    dd = bn.word_emb(word)
