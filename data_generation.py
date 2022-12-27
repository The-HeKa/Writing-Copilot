from transformers import AutoTokenizer, AutoModelForMaskedLM
import opencc
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import random

converter = opencc.OpenCC('s2tw.json')

ws = WS("./data", disable_cuda=False)
pos = POS("./data", disable_cuda=False)
ner = NER("./data", disable_cuda=False)

# cn_version
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")


# generate label
def generate_label(sentence,tag_pos,p):

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

    # print(sentence_list[0])
    # print(sent)

    return sent, label
