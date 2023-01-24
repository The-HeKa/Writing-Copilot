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

# -----
def word(sentence,tag_pos,_p):

    '''
    Inputs:
        label(['傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'],'D',0.5)

    Outputs
        (['傅达仁今运行安乐死，爆出自己20年前遭纬来体育台封杀，他懂自己哪里得罪到电视台。', '傅达仁今运行安乐死，爆出自己20年前遭纬来体育台封杀，他懂自己哪里得罪到电视台。'], 
        ['000010000010000000000000000010000000000', '000000000000000000000000000000000000000'], 
        [['傅达仁今MASK运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。'], 
         ['傅达仁今将运行安乐死，MASK突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。'], 
         ['傅达仁今将运行安乐死，却MASK爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。', '傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。'], 
         ['傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他MASK懂自己哪里得罪到电视台。', '傅达仁今将运行安乐死，却突然爆出自己20年前遭纬来体育台封杀，他不懂自己哪里得罪到电视台。']])
    '''

    sents = []
    labels = []

    try:
        sentence = converter.convert(sentence[0])
    except:
        sentence = converter.convert(sentence[0][0])
 
    sentence_list = [sentence]
    word_sentence_list = ws(sentence_list)
    pos_sentence_list = pos(word_sentence_list)

    label_list = []

    for i in range(len(word_sentence_list)):
        for r in range(len(word_sentence_list[i])):
            word_ = word_sentence_list[i][r]
            pos_ = pos_sentence_list[i][r]

            label_list.append([word_,pos_])
    sent = ''.join([i[0] for i in label_list])
    # get tag's next word's position
    def get_key_label(val):
        result = []
        position = 0
        del_list = []
        for i in range(len(label_list)):
            key, value = label_list[i]
            position += len(key)
            
            if val == value:

                # print(len(key),position)
                result.append([len(key),position])
                
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
                token = converter_.convert(sent.replace('MASK', sentence[start_pos:end_pos+1]))
                result.append([sent, token])
     
        return result

    mask = get_key_mask(tag_pos)

    for p in _p:
        sent = converter_.convert(sent)

        label_tag, _sent = get_key_label(tag_pos)
        for x in range(len(label_tag)):
            p_ = random.randint(1,100)/100
            if p_ <= p:
                # 保留
                pass
            else:
                # 刪除
                for _ in range(label_tag[x][0]):
                    _sent = list(_sent)
                    _sent[label_tag[x][1]-_-1] = '|'
                    _sent = ''.join(_sent)

        for k in range(len(label_tag)):
            x_ = _sent[:label_tag[k][1]]
            count = 0
            for d in x_:
                if d == '|': count +=1
            label_tag[k][1] -= count

        _sent = _sent.replace('|','')
        label_tag = [j[1] for j in label_tag]
                
        # 先將_sent編碼成0陣列，再根據label_tag把部分的0換成1
        encoded_str = tokenizer(_sent, padding=True, truncation=True) 
        tokens = tokenizer.convert_ids_to_tokens(encoded_str.input_ids)
        lengh = len(tokens[1:-1])
        label = ''
        while len(label)<lengh:
            label+='0'
        label = list(label)
        for i in label_tag:
            label[i] = '1'
        label = ''.join(label)

        labels.append(label)
        sents.append(_sent)

    return sents, labels, mask


# -----
def chenyu(sent, chenyu, p):
    '''
    Inputs:
        chenyu('细致地挨门逐户去调查访问。','挨门逐户', 0.1)
    
    Ouputs:
        (['细致MASK查访问。', '细致地挨门逐户去调查访问。'], ['细致查访问。', '001000'])
    '''
    mask = ''
    label = [] 

    front_random = random.randint(0,2)
    back_random = random.randint(0,2)
    chenyu_pos = sent.find(chenyu)
    tag = sent[chenyu_pos-front_random:chenyu_pos+back_random+len(chenyu)+1]
    mask = [sent.replace(tag,'MASK'), sent]

    p_ = random.randint(1,100)/100
    if p_ <= p:
        # 保留
        label_tag = chenyu_pos+back_random+len(chenyu)
    else:
        # 刪除
        sent = mask[0].replace('MASK','')
        label_tag = mask[0].find('MASK')

    encoded_str = tokenizer(sent, padding=True, truncation=True) 
    tokens = tokenizer.convert_ids_to_tokens(encoded_str.input_ids)
    lengh = len(tokens[1:-1])
    label = ''
    while len(label)<lengh:
        label+='0'
    label = list(label)
    label[label_tag] = '1'
    label = ''.join(label)

    return mask, [sent,label]


if __name__ == '__main__':
    print(word(['傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。'],'D',[0.1,0.9]))
    print(chenyu('细致地挨门逐户去调查访问。','挨门逐户', 0.1))