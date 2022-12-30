from transformers import AutoTokenizer, AutoModelForMaskedLM
import opencc
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import random
import os
from line_profiler import LineProfiler


converter = opencc.OpenCC('s2tw.json')
converter_ = opencc.OpenCC('tw2sp.json')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ws = WS("./data", disable_cuda= False)
# pos = POS("./data",disable_cuda= False)

ws = WS("./data")
pos = POS("./data")

# cn_version
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


# generate label
def label(sentence,tag_pos,p):

    # print(sentence)
    sentence = converter.convert(sentence[0])
    sentence_list = [sentence]
    # print(sentence_list[0])
    word_sentence_list = ws(sentence_list)
    pos_sentence_list = pos(word_sentence_list)

    label_list = []

    for i in range(len(word_sentence_list)):
        for r in range(len(word_sentence_list[i])):
            word_ = word_sentence_list[i][r]
            pos_ = pos_sentence_list[i][r]

            label_list.append([word_,pos_])

    # get tag's next word's position
    def get_key(val):
        result = []
        position = 0
        del_list = []
        for i in range(len(label_list)):
            key, value = label_list[i]
            position += len(key)
            
            if val == value:

                # judge del or save
                # p = 0.9
                p_ = random.randint(1,100)/100
                if p_ <= p:
                    # del
                    position -= len(key)
                    result.append([0,position])
                    del_list.append(i)
                    # print(0,position, sentence_list[0][position])
                else:
                    # save
                    result.append([1,position])
                    # print(1,position, sentence_list[0][position])
        
        # print("".join([x[0] for x in label_list]))
        del_list.sort(reverse=True)
        for _ in del_list:
            del label_list[_]
        sent = "".join([x[0] for x in label_list])
                    
        return result, sent

    label_tag, sent = get_key(tag_pos)
    encoded_str = tokenizer(sent, padding=True, truncation=True) 
    tokens = tokenizer.convert_ids_to_tokens(encoded_str.input_ids)

    lengh = len(tokens[1:-1])
    label = ''

    while len(label)<lengh:
        label+='0'


    for i in [r[1] for r in label_tag]:
        label = list(label)
        # print(label)
        label[i] = '1'
        label = ''.join(label)
    # converter.set_conversion('tw2sp')
    sent = converter_.convert(sent)
    # print(sentence_list[0])

    return sent, label

def mask(sentence_,tag_pos):

    ''''
    Inputs:
    generate_mask(['傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'],'D')

    Outputs:
    [['傅达仁今MASK运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '将'],
     ['傅达仁今将运行安乐死，MASK突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '却'],
     ['傅达仁今将运行安乐死，却MASK爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '突然'],
     ['傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他MASK懂自己哪里得罪到电视台。', '不']]
    '''
    
    # print(sentence)
    sentence = converter.convert(sentence_[0])
    sentence_list = [sentence]
    word_sentence_list = ws(sentence_list)
    pos_sentence_list = pos(word_sentence_list)

    label_list = []
    result_list = []

    for i in range(len(word_sentence_list)):
        for r in range(len(word_sentence_list[i])):
            word_ = word_sentence_list[i][r]
            pos_ = pos_sentence_list[i][r]

            label_list.append([word_,pos_])

    # get tag's next word's position
    def get_key(val):
        result = []
        position = 0
        for i in range(len(label_list)):
            key, value = label_list[i]
            position += len(key)
            
            if val == value:
                start_pos = position - len(key)
                end_pos = position-1

                sent = converter_.convert(sentence[:start_pos]+'MASK'+sentence[end_pos+1:])
                token = converter_.convert(sentence[start_pos:end_pos+1])
                result.append([sent, token])
                # print([sent, token])
     
        return result

    return get_key(tag_pos)
