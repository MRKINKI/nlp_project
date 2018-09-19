import re


class Rule:
    def __init__(self):
        self.re_title = '(文章|正文|本文|新闻).*(内容|主旨|主题|大意|说了|介绍|讲了)'
    
    def match_titie(self, sample):
        title = sample['article_title']
        for qa_sample in sample['questions']:
            question = qa_sample['question']
            if re.match(self.re_title, question):
                qa_sample['pred'] = title
