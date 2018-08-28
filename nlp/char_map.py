# -*- coding: utf-8 -*-
import re

class CharMap:
    def __init__(self):
        self.adict = {'':'',
                     '％':'%',
                     '<content>':'',
                     '</content>':''}
        self.adict.update(self.char_map(65345, 97, 26))
        self.adict.update(self.char_map(65313, 65, 26))
        self.adict.update(self.char_map(65296, 0, 10, 'str'))
                     
    def char_map(self, a_start, b_start ,num, method = 'chr'):
        map_dict = {}
        for i,j in zip(range(a_start,a_start + num), range(b_start, b_start + num)):
            if method == 'chr':
                map_dict[chr(i)] = chr(j)
            elif method == 'str':
                map_dict[chr(i)] = str(j)
        return map_dict

    def multiple_replace(self, text):
        for a, b in self.adict.items():
            text = re.sub(a,b,text)
        return text



