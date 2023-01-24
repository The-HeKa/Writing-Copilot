from itertools import product
from tqdm import tqdm
import csv
from data_generation import word, chenyu
import opencc
import os

sents_file = 'sents.csv'
sents_ = []
with open(sents_file,'r') as f:
    sents_ = [i for i in csv.reader(f)][1:]

sents = sents_[:200000]

def get_len(a):
    return(len(a))

def word_data(sents:list, len:list, p:list, pos_tag:str):
    file_name = 'word_'+str(get_len(sents))
    if not os.path.isdir(file_name):
        os.mkdir(file_name)

    sents_ = []
    mask = []
    label = []

    for sent in tqdm(sents):
        try:
            sent = [sent[:max(len)]]
            a, b, c = word(sent, pos_tag, p)
            label.append([a,b])
            mask += c
            del a
            del b
            del c
        except:
            pass
    
    # save dict
    label_dict = {}
    for r in range(get_len(p)):
        for y in len:
            label_dict['label_r_p'.replace('r',str(p[r])).replace('p',str(y))] = []

    for i in range(get_len(sents)):
        for r in range(get_len(p)):
            for y in len:
                try:
                    label_b = label[i][1][r][:y]
                    label_b = list(label_b)
                    while get_len(label_b) < y:
                        label_b += '0'
                    label_b = ''.join(label_b)
                    if '1' in label_b:
                        label_dict['label_r_p'.replace('r',str(p[r])).replace('p',str(y))].append([label[i][0][r][:y],label_b])
                except:
                    pass
                    # print('error')
                # print(label_dict['label_r_p'.replace('r',str(p[r])).replace('p',str(y))])

    for i in label_dict.keys():
        with open(file_name+'/'+i+'_'+pos_tag+'.csv',mode='w') as f:
            wt = csv.writer(f)
            for r in label_dict[i]:
                wt.writerow([r[0]]+list(r[1]))
    
    # save mask
    for k in range(get_len(mask)):
        for i in len:
            with open(file_name+'/'+'mask_'+str(i)+'.csv',mode='a') as f:
                wt = csv.writer(f)
                if 'MASK' in mask[k][0][:i]:
                    wt.writerow([mask[k][0][:i],mask[k][1][:i]])

word_data(['傅达仁今将运行安乐死，突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。'], [64,32], [0.5,0.8], 'D')

lengths = [96,72,64,32,16]
ps = [0.8,0.5,0.2]
tags = ['A','D','P','Nf','Cbb']

for i in tags:
    word_data(sents,lengths,ps,i)