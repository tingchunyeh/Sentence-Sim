import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob
import collections
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

### Loading data ###
def read_raw_cData(data_path):
    f = open(data_path, errors='ignore')
    questions = []
    for line in tqdm(f, desc="{reading raw customer data ...}"):
        questions.append(line[:-1])
    questions = pd.Series(questions)
    raw_data = pd.DataFrame()
    raw_data['用户问题'] = questions
    raw_data['question_len'] = raw_data['用户问题'].apply(lambda x: len(x.strip()))

    return raw_data

def read_csvFiles(data_path):
    csv_files = glob.glob(data_path)
    data = pd.DataFrame()
    for csv_file in tqdm(csv_files, desc="{Reading csv files ...}"):
        data = pd.concat([data, read_csvFile(csv_file)])
    
    return data

def read_csvFile(csv_file):
    name = csv_file.split("/")[-1].split('.')[0]
    f = open(csv_file, encoding='gb18030', errors='ignore')
    data = pd.read_csv(f, low_memory=False).iloc[:, 0:6]
    if data.shape[1]==5:
        data.loc[:, '人工标注'] = pd.Series([np.nan for i in range(data.shape[0])], index=data.index)
    data.columns = ['用户问题', '问题分词', '分类', '标准问', '匹配分数', '人工标注']
    data.loc[:,'csv_name'] = pd.Series([name for i in range(data.shape[0])], index=data.index)
    
    return data

def combine_kbase(data, kbase_data):
    kbase_data.loc[:, 'kbase'] =  pd.Series([True for i in range(kbase_data.shape[0])], index=kbase_data.index)
    data.loc[:, 'kbase'] =  pd.Series([False for i in range(data.shape[0])], index=data.index)
    data = pd.concat([data, kbase_data], sort=False)
    data.fillna(np.nan)
    return data
    
def normalize_data(data):
    # convert type
    print("converting type ...")
    if '用户问题' in data.columns:  data.loc[:,'用户问题'] = data['用户问题'].astype(str)
    if '标准问' in data.columns:  data.loc[:,'标准问'] = data['标准问'].astype(str)
    if '问题分词' in data.columns: data.loc[:,'问题分词'] = data['问题分词'].astype(str)
    
    # calculating question length
    if '用户问题':
        print("calculating question length ...")
        data.loc[:, 'question_len'] = data['问题分词'].apply(lambda x: len("".join(x.split(" "))))
    
    print("calculating tokens length ...")
    data.loc[:, 'tokens_len'] = data['问题分词'].apply(lambda x: len(x.split(" ")))
    
    # drop duplicate
    print("Dropping duplicates ...")
    print("original # data {}".format(data.shape[0]))
    data = data.drop_duplicates(subset=['问题分词'], keep='last').reset_index(drop=True) # keep last duplicate
    data = data.dropna(subset=['分类'])
    print("after dropping, # data {}".format(data.shape[0]))

    return data
    
### others ###    

def words_count(data):
    # words frequency
    words = []
    for i in tqdm(data.index):
        words.extend(data.loc[i, '问题分词'].strip().split())
    words_count = collections.Counter(words)
    print("total number of words: ",len(words_count))

    words_countDF = pd.DataFrame.from_dict(words_count, orient='index').reset_index()
    words_countDF.columns = ["word", "freq"]
    return words_countDF
    
    
    
### Plot ###
def cat_barh(data, by = 'csv_name', figsize=(10, 20)):
    # category distribution 
    plt.figure(figsize=figsize)
    ax = data[by].value_counts().plot(kind='barh')
    for p in ax.patches:    
        ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')
    plt.yticks(fontsize = 16)
    plt.show()

def score_box(data, by='csv_name'):
    plt.figure(figsize=(16, 6))
    keys = data.groupby(by)['匹配分数'].median().sort_values().keys()
    ax = sns.boxplot(x=by, y='匹配分数', data=data, order=keys)
    plt.title('匹配分数 for each {}'.format(by), fontsize=14)
    plt.xticks(fontsize = 14, rotation=90)
    plt.show()

def questionLen_score_box(data):
    plt.figure(figsize=(14, 6))
    plt.title('matching score boxplot for different question_length', fontsize=14)
    ax = sns.boxplot(x="question_len", y="匹配分数", data=data[data.question_len<=50])
    ax.set_xlabel('question_len', fontsize=14)
    ax.set_ylabel('匹配分数', fontsize=14)
    plt.show()


def question_len_dist(data):
    plt.figure(figsize=(14, 6))
    data_amount_byLen = data[data.question_len<=50].groupby('question_len').size()
    plt.title('# questions for different question_length', fontsize=14)
    ax = data_amount_byLen.plot(kind='bar', )
    ax.set_xlabel('question_len', fontsize=14)
    ax.set_ylabel('# questions', fontsize=14)
    count = 0
    tot = data_amount_byLen.sum()
    for i, p in enumerate(ax.patches):    
        if (i+1) not in data_amount_byLen.index: continue
        count += data_amount_byLen.loc[i+1]
        ax.annotate(str(int(round(float(count)/tot,2)*100)), xy=(p.get_x(), p.get_height()+50), size=14 )
    plt.show()
    
def tokens_len_dist(data):
    plt.figure(figsize=(14, 6))
    data_amount_byLen = data[data.question_len<=50].groupby('tokens_len').size()
    plt.title('# questions for different tokens_length', fontsize=14)
    ax = data_amount_byLen.plot(kind='bar', )
    ax.set_xlabel('tokens_len', fontsize=14)
    ax.set_ylabel('# questions', fontsize=14)
    count = 0
    tot = data_amount_byLen.sum()
    for i, p in enumerate(ax.patches):    
        if (i+1) not in data_amount_byLen.index: continue
        count += data_amount_byLen.loc[i+1]
        ax.annotate(str(int(round(float(count)/tot,2)*100)), xy=(p.get_x(), p.get_height()+50), size=14 )
    plt.show()

def question_len_boxplot(data):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data.question_len)
    plt.show()
    print( data.question_len.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) )
    
def words_freq_plot(words_countDF):
    plt.figure(figsize=(10, 5))
    sns.boxplot(words_countDF.freq)
    plt.title('word frequency', fontsize = 16 )
    plt.show()
    print( words_countDF.freq.quantile([0.05, 0.25, 0.5, 0.675, 0.75, 0.95]) )

    freq_dist = words_countDF.groupby('freq').word.count()
    freq_dist = freq_dist[freq_dist.keys()<150]
    plt.figure(figsize=(14,6))
    plt.title('word frequency', fontsize=14)
    plt.plot(freq_dist.keys(), freq_dist.values)
    plt.xlabel('frequency', fontsize=14)
    plt.ylabel('# words', fontsize=14)
    plt.show()
    
### Data preprocess ###

def remove_rareWords(data, n=5):
    words_countDF = words_count(data)
    rare_wordsSet = set(words_countDF[words_countDF.freq<n].word)
    data.loc[:, '问题分词'] = data.loc[:, '问题分词'].apply(lambda x: " ".join([w if w not in rare_wordsSet else "" for w in x.split(" ")]).strip() )
    data.loc[:, 'question_len'] = data.loc[:, '问题分词'].apply(lambda x: len(x.split(" ")) if x != ""  else 0)
    data = data[ data['question_len']!=0 ]
    print("# words after removal: {}".format(words_countDF.shape[0]-len(rare_wordsSet)))
    
    return data

def remove_rareCats(data, n=100):
    cat_size = data.groupby('分类').size()
    rare_cat = cat_size[cat_size<n].index
    data = data[~data['分类'].isin(rare_cat)]
    
    return data

def output_train_test_data(data, cat='csv_name', test_size=0.2, seed=42):
    # prepare training, validation dataset
    X_train, X_test, y_train, y_test = train_test_split(data, data.loc[:, cat], test_size=test_size, random_state=seed)
    
    # output train.txt
    with open("train.txt", "w") as text_file:
        for i in tqdm(X_train.index, desc="{Preparing train.txt ...}"):        
            text_file.write(X_train.loc[i, '问题分词'].strip()+"\t__label__"+y_train.loc[i,]+"\n")

    # output test.txt
    with open("test.txt", "w") as text_file:
        for i in tqdm(X_test.index, desc="{Preparing test.txt ...}"):        
            text_file.write(X_test.loc[i, '问题分词'].strip()+"\t__label__"+y_test.loc[i,]+"\n")
            
    return X_train, X_test, y_train, y_test 


### Evaluation ###

def load_testSet(test_file):
    # evaluate
    ftest = open(test_file, "r", encoding="utf-8")
    tests = []
    label = []
    for i, t in enumerate(ftest):
        data = t.strip().split('\t')
        tests.append(data[0])
        label.append(data[1])
    ftest.close()
    return tests, label

def evaluation_DF(precision, recall, fscore, support, labels):
    sort_index = np.argsort(fscore)[::-1]
    precision = precision[sort_index]
    recall = recall[sort_index]
    fscore = fscore[sort_index]
    support = support[sort_index]
    labels = labels[sort_index]

    res = pd.DataFrame()
    res['labels'] = labels
    res['support'] = support
    res['fscore'] = fscore
    res['precision'] = precision
    res['recall'] = recall
    
    return res