import logging
import numpy as np
import torch
import pandas as pd
# from torch.utils.data import Dataset, DataLodaer
from torch.autograd import Variable
from tqdm import tqdm
import time
from gensim.models import Word2Vec, KeyedVectors
import os
import collections

def get_batch(batch, wv, default_wv, dropout=0.0):
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))
    
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            if np.random.rand()<dropout: continue
            if batch[i][j] not in wv:
                embed[j, i, :] = default_wv
            else:
                embed[j, i, :] = wv[batch[i][j]]
    return torch.from_numpy(embed).float(), lengths

def create_pair_data(kb_answer):
    pairs = []
    time.sleep(1)
    for s_id in tqdm(kb_answer.index):
        sentence, code, code2 = kb_answer.loc[s_id, 'question'], kb_answer.loc[s_id, 'code'], kb_answer.loc[s_id, 'code2']
        same_code_sentences = kb_answer[kb_answer.code == code]
        same_code2_sentences = kb_answer[(kb_answer.code2 == code2) & (kb_answer.code!=code)]
        
        
        # random sample at most 5 sentences in same code
        same_code_samples = same_code_sentences.sample(n=min(20,same_code_sentences.shape[0]))
        for s2 in same_code_samples['question']:
            pairs.append((sentence, s2, 5))
            
        # random sample at most 5 sentences in same code2
        if same_code2_sentences.shape[0]>0:
            same_code2_samples = same_code2_sentences.sample(n=min(20,same_code2_sentences.shape[0]))
            for s2 in same_code2_samples['question']:
                pairs.append((sentence, s2, 3))
            
        # random sample at most 10 sentence in different code2
        different_samples = kb_answer[(kb_answer.code!=code) & (kb_answer.code2!=code2)].sample(n=20)
        for s2 in different_samples['question']:
            pairs.append((sentence, s2, 0))
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df.columns = ['s1', 's2', 'score']
    return pairs_df

def read_kb(data_path):
    kb = pd.read_csv(os.path.join(data_path,'kb_parsed_tokenized.csv'))
#     ha = pd.read_csv(data_path+'/qa_history.csv').dropna(subset=['参考答案分词', '坐席回复分词', '坐席回复'], how='any')
    kb['answer'] = pd.Categorical(kb['answer'])
    kb['code'] = kb['answer'].cat.codes
    kb['category'] = pd.Categorical(kb['category'])
    kb['code2'] = kb['category'].cat.codes
    kb = kb[['question', 'code', 'code2']]
    print("original shape: ", kb.shape)
    print("dropping duplicates ...")
    kb = kb.drop_duplicates(subset= ["question"])
    kb = kb.dropna()
    print("after dropping duplicates: ", kb.shape)
    return kb

def get_word_dict(sentences):
    # create vocab of dict
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict

def get_wordVec(word_dict, wordVec):
    word_vec = {}
    with open(wordVec, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(" ", 1)
            vec = vec[:-1]
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found ({0}/{1}) words with w2v vectors'.format(len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, wordVec):
    word_dict = get_word_dict(sentences)
    wv = get_wordVec(word_dict, wordVec)
    print("vocab size: {}".format(len(word_dict)))
    default_wv = np.zeros(300)
    for key, value in wv.items():
        default_wv += value
    default_wv /= len(wv)
    return wv, default_wv


def get_nl(data_path):
    s1 = {}
    s2 = {}
    target = {}
    
    dico_label = {'0': 0,  '3': 1, '5': 2}
    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        file = os.path.join(data_path, data_type+".csv")
        print("processing {} ...".format(file))
        s1[data_type]['sent']= np.array([line.rstrip().split(",")[0] for line in open(file , mode='r', encoding='utf-8-sig')])
        s2[data_type]['sent']= np.array([line.rstrip().split(",")[1] for line in open(file , mode='r', encoding='utf-8-sig')])
        target[data_type]['data']= np.array([ dico_label[line.rstrip().split(",")[2]] for line in open(file , mode='r', encoding='utf-8-sig')])
        
        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == len(target[data_type]['data'])
        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))
    
    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],'label': target['test']['data']}
    return train, dev, test



