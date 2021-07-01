import logging
import json
import os
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import cossim
import pandas as pd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TRAIN_ITERS = 300

def getCorpus(df, data_type):
    corpus_list = []
    df['summary_processed'] = df['summary_processed'].astype(str)
    df['description_processed'] = df['description_processed'].astype(str)
    df['message_processed'] = df['message_processed'].astype(str)
    df['processDiffCode'] = df['processDiffCode'].astype(str)

    if data_type == 'text':
        for index, row in df.iterrows():
            sentence = row['summary_processed'] + row['description_processed'] + row['message_processed']
            words = sentence.split(" ")
            sentence_segment = []
            for word in words:
                if word.strip() != '':
                    sentence_segment.append(word.strip())
            corpus_list.append(sentence_segment)
    else:
        for index, row in df.iterrows():
            sentence = row['processDiffCode']
            words = sentence.split(" ")
            sentence_segment = []
            for word in words:
                if word.strip() != '':
                    sentence_segment.append(word.strip())
            corpus_list.append(sentence_segment)
            
    return corpus_list

# create data frame of all project's data
df = pd.DataFrame()
files = os.listdir('data/')
for file in files:
    tmp_df = pd.read_parquet('data/'+file)
    textual_data = tmp_df.loc[tmp_df['train_flag'] == 1]
    df = df.append(textual_data, ignore_index=True)
    
# creating data set for code and text
text_dataset = getCorpus(df, data_type='text')
code_dataset = getCorpus(df, data_type='code')
code_dct = Dictionary(code_dataset)
text_dct = Dictionary(text_dataset)
code_corpus = [code_dct.doc2bow(line) for line in code_dataset]  # convert corpus to BoW format
text_corpus = [text_dct.doc2bow(line) for line in text_dataset]  # convert corpus to BoW format
# create Code mode and Text model
code_model = TfidfModel(code_corpus)
code_model.save("tfidf/models/code_tfidf.model")
text_model = TfidfModel(text_corpus)
text_model.save("tfidf/models/text_tfidf.model")


def read_data(path):
    res = []
    filepath = path
    logging.info("Loaded the file:"+filepath)
    if os.path.isfile(filepath):
        file = open(filepath, 'rb')
        testlist = json.loads(file.read())
        res.extend(testlist)
        file.close()
    return res


def getSim(vec1, vec2):
    return cossim(vec1, vec2)


def getTextSim(commitText, issueText):
    res = 0
    for cText in commitText:
        cVec = text_model[text_dct.doc2bow([cText])]
        for iText in issueText:
            iVec = text_model[text_dct.doc2bow([iText])]
            res = max(res, getSim(cVec, iVec))
    return res


def getCodeSim(commitCode, issueCode):
    cVec = code_model[code_dct.doc2bow(commitCode)]
    iVec = code_model[code_dct.doc2bow(issueCode)]
    return getSim(cVec, iVec)


def learn(list, ITR):
    ThresVal = 0.0
    Step = 0.01
    LThres = 0.0
    F = 0.0
    RMax = ITR
    while ThresVal <= 1:
        TP = 0
        FP = 0
        FN = 0
        for link in list:
            if link['val'] >= ThresVal:
                if link['type'] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if link['type'] == 1:
                    FN += 1
        precision = TP/(TP+FP+1e-8)
        recall = TP/(TP+FN+1e-8)
        f_measure = (2*precision*recall)/(precision + recall+1e-8)
        if recall >= ITR:
            if (f_measure > F) or (f_measure == F and recall > RMax):
                LThres = ThresVal
                RMax = recall
                F = f_measure
        ThresVal = ThresVal + Step
    return LThres


def getRes(test_set, t):
    size = len(test_set)
    right = 0.0
    for link in test_set:
        if link['type'] == 1 and link['val'] >= t:
            right += 1
        elif link['type'] == 0 and link['val'] < t:
            right += 1
    return right/size


def evaluation(test_set, t):
    TP = 0
    FP = 0
    FN = 0
    for link in test_set:
        if link['val'] >= t:
            if link['type'] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if link['type'] == 1:
                FN += 1
    precision = float(TP) / (TP + FP+1e-8)
    recall = float(TP) / (TP + FN+1e-8)
    f_measure = (2 * precision * recall) / (precision + recall+1e-8)
    logging.info("precision:%f  recall:%f  f_measure:%f" % (precision, recall, f_measure))

    
def build():
    filelist = os.listdir('data/')
    for i in range(0, len(filelist)):
        filepath = os.path.join('data/', filelist[i])
        logging.info("Loaded the file:" + filepath)
        if os.path.isfile(filepath):
            df1 = pd.read_parquet(filepath)
            df1 = df1.loc[df1['train_flag'] == 1]

            df1['summary_processed'] = df1['summary_processed'].astype(str)
            df1['description_processed'] = df1['description_processed'].astype(str)
            df1['message_processed'] = df1['message_processed'].astype(str)
            df1['processDiffCode'] = df1['processDiffCode'].astype(str)
    
            link_list = []
            for index, row in df1.iterrows():
                type = row['label'] 
                val = max(getTextSim((row['summary_processed']+row['description_processed']).split(" "), row['message_processed'].split(" ")),
                          getCodeSim(row['description_processed'].split(" "), row['processDiffCode'].split(" "))) 
                link_list.append({'type': type, 'val': val})
            res = json.dumps(link_list, indent=4)
            trainSet = open('train/'+filelist[i].split('.')[0]+'.dat', "w")
            trainSet.write(res)
            trainSet.close()
            
def build_test():
    filelist = os.listdir('data/')
    for i in range(0, len(filelist)):
        filepath = os.path.join('data/', filelist[i])
        logging.info("Loaded the file:" + filepath)
        if os.path.isfile(filepath):
            df1 = pd.read_parquet(filepath)
            df1 = df1.loc[df1['train_flag'] == 0]

            df1['summary_processed'] = df1['summary_processed'].astype(str)
            df1['description_processed'] = df1['description_processed'].astype(str)
            df1['message_processed'] = df1['message_processed'].astype(str)
            df1['processDiffCode'] = df1['processDiffCode'].astype(str)
    
            link_list = []
            for index, row in df1.iterrows():
                type = row['label'] 
                val = max(getTextSim((row['summary_processed']+row['description_processed']).split(" "), row['message_processed'].split(" ")),
                          getCodeSim(row['description_processed'].split(" "), row['processDiffCode'].split(" "))) 
                link_list.append({'type': type, 'val': val})
            res = json.dumps(link_list, indent=4)
            trainSet = open('test/'+filelist[i].split('.')[0]+'.dat', "w")
            trainSet.write(res)
            trainSet.close()

            
def main():
    build()
    build_test()
    files = os.listdir('train/')
    for file in files:
        trainset = read_data(path='train/'+file)
        testset = read_data(path='test/'+file)
        t = learn(trainset, 0.88)
        res = getRes(testset, t)
        logging.info(t)
        logging.info(res)
        evaluation(testset, t)
        logging.info("Finished!")

if __name__ == "__main__":
    main()