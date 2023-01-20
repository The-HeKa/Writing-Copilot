from transformers import AutoTokenizer
import opencc
from ckiptagger import WS, POS
import random
import os


converter = opencc.OpenCC('s2tw.json')
converter_ = opencc.OpenCC('tw2sp.json')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ws = WS("./data", disable_cuda= False)
# pos = POS("./data",disable_cuda= False)

ws = WS("./data")
pos = POS("./data")

# cn_version
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


# generate mask and label
def word(sentence,tag_pos,p):

    '''
    Inputs:
        label(['傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'],'D',0.5)

    Outputs
        ('傅达仁今运行安乐死，爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', 
        '0000100000100000000000000000010000000000', 
        [['傅达仁今MASK运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '将'], 
        ['傅达仁今将运行安乐死，MASK突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '却'], 
        ['傅达仁今将运行安乐死，却MASK爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '突然'], 
        ['傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他MASK懂自己哪里得罪到电视台。', '不']])
    '''

    sentence = converter.convert(sentence[0])
    sentence_list = [sentence]
    word_sentence_list = ws(sentence_list)
    pos_sentence_list = pos(word_sentence_list)

    label_list = []

    for i in range(len(word_sentence_list)):
        for r in range(len(word_sentence_list[i])):
            word_ = word_sentence_list[i][r]
            pos_ = pos_sentence_list[i][r]

            label_list.append([word_,pos_])

    # get tag's next word's position
    def get_key_label(val):
        result = []
        position = 0
        del_list = []
        for i in range(len(label_list)):
            key, value = label_list[i]
            position += len(key)
            
            if val == value:

                # judge del or save
                p_ = random.randint(1,100)/100
                if p_ <= p:
                    # del
                    position -= len(key)
                    result.append([0,position])
                    del_list.append(i)
                else:
                    # save
                    result.append([1,position])
        
        del_list.sort(reverse=True)
        for _ in del_list:
            del label_list[_]
        sent = "".join([x[0] for x in label_list])
                    
        return result, sent
    
    # get tag's next word's position
    def get_key_mask(val):
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
     
        return result

    mask = get_key_mask(tag_pos)
    label_tag, sent = get_key_label(tag_pos)
    
    encoded_str = tokenizer(sent, padding=True, truncation=True) 
    tokens = tokenizer.convert_ids_to_tokens(encoded_str.input_ids)

    lengh = len(tokens[1:-1])
    label = ''

    while len(label)<lengh:
        label+='0'


    for i in [r[1] for r in label_tag]:
        label = list(label)
        label[i] = '1'
        label = ''.join(label)

    sent = converter_.convert(sent)

    return sent, label, mask

print(word(['傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'],'D',0.5))