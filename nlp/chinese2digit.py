# -*- coding: utf-8 -*-
import re

numerals ={'零':0, '一':1, '二':2, '两':2, '三':3, '四':4, 
           '五':5, '六':6, '七':7, '八':8, '九':9, '十':10, 
           '百':100, '千':1000, '万':10000, '亿':100000000}      
numeral_amounts = { '十':10, '百':100, '千':1000, '万':10000, '亿':100000000}
prefix_non_used_numerals = {'几', '数', '.'}
for i in range(10):
    prefix_non_used_numerals.add(str(i))
postfix_non_used_numerals = {'几', '多'}

def chinese2digits(uchars_chinese):
    total = 0
    r = 1              
    for i in range(len(uchars_chinese) - 1, -1, -1):
        val = numerals.get(uchars_chinese[i])
        if val >= 10 and i == 0: 
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total
    
def is_contain(m, start_idx, end_idx):
    for key in numeral_amounts:
        if key in m[start_idx: end_idx]:
            return True
    return False
    
def duplicate(m):
    n = []
    for term in m:
        if not len(n):
            n.append(term)
        elif "@@" in term and term != n[-1]:
            n.append(term)
        elif "@@" not in term:
            n.append(term)
    n = ''.join([re.sub("@@", '', t) for t in n])
    return n

def num_convert(text):
    t_list = list(text)
    idx = 0
    while idx < len(t_list):
        if t_list[idx] in numerals:
            jdx = idx +  1
            k = 0
            while jdx < len(t_list):
                if t_list[jdx] in numerals:
                    jdx += 1
                else:
                    break
            # 一三
            if jdx - idx - 1 > 0 and not is_contain(t_list, idx, jdx):
                k = 1
            elif jdx < len(t_list):
                if t_list[jdx] in postfix_non_used_numerals:
                    k = 1
            if idx > 0:
                if t_list[idx - 1] in prefix_non_used_numerals:
                    k = 1
            if k == 0:
                num_text = ''.join(t_list[idx: jdx])
                num = chinese2digits(num_text)
                num_list = ['@@' + str(num)] * (jdx - idx)
                t_list = t_list[:idx] + num_list + t_list[jdx:]
            idx = jdx
        else:
            idx += 1
    convert_text = duplicate(t_list)
    return convert_text
            

if __name__ == '__main__':   
    print(chinese2digits('三十') )
    text = '长江三号在二十年前下水，五成人七'
    text = '款无人机是耗资相当于14.6亿元人民币'
    aa = num_convert(text)
    print(aa)
