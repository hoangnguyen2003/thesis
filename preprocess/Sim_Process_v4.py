import numpy as np
import pandas as pd
import torch
import json
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mosi_len_pos = None
mosi_len_neu = None
mosi_len_neg = None

def mosi_mosei(path):
    positives, neutrals, negatives = [], [], []
    data = pd.read_csv(path)
    for i in range(data.shape[0]):
        line = data.iloc[i]
        vid = line['video_id']
        clip_id = line['clip_id']
        text = line['text'].strip().lower()
        label = line['label']
        annotation = line['annotation'].lower()
        if annotation == 'negative':
            negatives.append([str(vid)+'_'+str(clip_id), text, annotation, label])
        elif annotation == 'positive':
            positives.append([str(vid)+'_'+str(clip_id), text, annotation, label])
        else:
            neutrals.append([str(vid)+'_'+str(clip_id), text, annotation, label])

    return positives, neutrals, negatives

def iemocap(path):
    positives, neutrals, negatives = [], [], []
    data = pd.read_csv(path)
    for i in range(data.shape[0]):
        line = data.iloc[i]
        label = line['label']
        vid_cid = line['vid_cid']
        text = line['text']
        if label == 'neu':
            neutrals.append([vid_cid, text, label])
        elif label in ['hap', 'exc']:
            positives.append([vid_cid, text, label])
        else:
            negatives.append([vid_cid, text, label])

    return positives, neutrals, negatives

# polarity_set_2v(mosi_path, mosei_path, iemocap_path, meld_path)
def polarity_set_2v(path1, path2, path5, path_cur):
    positives, neutrals, negatives = [], [], []
    data2 = pd.read_csv(path_cur)

    for i in range(data2.shape[0]):
        line = data2.iloc[i]
        id = str(line['Dialogue_ID']) + '_' + str(line['Utterance_ID'])
        emotion = line['Emotion'].strip()
        sentiment = line['Sentiment'].strip()
        raw_text = line['Utterance']
        if sentiment == 'negative':
            negatives.append([id, raw_text, sentiment, emotion])
        elif sentiment == 'positive':
            positives.append([id, raw_text, sentiment, emotion])
        else:
            neutrals.append([id, raw_text, sentiment, emotion])

    mosi_pos, mosi_neu, mosi_negatives = mosi_mosei(path1)
    mosei_pos, mosei_neu, mosei_negatives = mosi_mosei(path2)
    iemocap_pos, iemocap_neu, iemocap_negatives = iemocap(path5)

    global mosi_len_pos
    mosi_len_pos = len(mosi_pos)
    global mosi_len_neg
    mosi_len_neg = len(mosi_negatives)

    mosi_pos.extend(mosei_pos)
    mosi_neu.extend(mosei_neu)
    mosi_negatives.extend(mosei_negatives)

    return (positives, neutrals, negatives), (mosi_pos, mosi_neu, mosi_negatives), (iemocap_pos, iemocap_neu, iemocap_negatives)

def cal_cosine_sim_2v(mosi_sen, meld_sen, iemocap_sen, mosi_len):
    mosi_len = len(mosi_sen)
    meld_len = len(meld_sen)
    iemocap_len = len(iemocap_sen)

    print('mosi + mosei len:{}, meld len:{}, iemocap len:{}'.format(mosi_len, meld_len, iemocap_len))

    cosine_sims_mosi2meld = np.zeros([mosi_len, meld_len])
    cosine_sims_mosi2iemocap = np.zeros([mosi_len, iemocap_len])
    cosine_sims_meld2mosi = np.zeros([meld_len, mosi_len])
    cosine_sims_iemocap2mosi = np.zeros([iemocap_len, mosi_len])

    mosi, meld, iemocap = [], [], []
    for sen in mosi_sen:
        mosi.append(sen[1])

    for sen in meld_sen:
        meld.append(sen[1])

    for sen in iemocap_sen:
        iemocap.append(sen[1])

    print('calculate the meld')
    embeddings_meld = []
    for j, s in enumerate(meld):
        inputs_meld = tokenizer(s, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddings = model(**inputs_meld, output_hidden_states=False, return_dict=True).pooler_output
            embeddings_meld.append(embeddings)

    print('calculate the mosi')
    embeddings_mosi = []
    for j, s in enumerate(mosi):
        inputs_mosi = tokenizer(s, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embedding = model(**inputs_mosi, output_hidden_states=False, return_dict=True).pooler_output
            embeddings_mosi.append(embedding)

    print('calculate the iemocap')
    embeddings_iemocap = []
    for j, s in enumerate(iemocap):
        input_iemocap = tokenizer(s, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embeddingss = model(**input_iemocap, output_hidden_states=False, return_dict=True).pooler_output
            embeddings_iemocap.append(embeddingss)

    embeddings_mosi = [e.cpu().numpy().flatten() for e in embeddings_mosi]
    embeddings_meld = [e.cpu().numpy().flatten() for e in embeddings_meld]
    embeddings_iemocap = [e.cpu().numpy().flatten() for e in embeddings_iemocap]

    print('calculate the simility of mosi to meld & iemocap')
    for i in tqdm(range(mosi_len)):
        for j in range(meld_len):
            cosine_sim_i_j = 1 - cosine(embeddings_mosi[i], embeddings_meld[j])
            cosine_sims_mosi2meld[i][j] = cosine_sim_i_j
        for j in range(iemocap_len):
            cosine_sim_i_j = 1 - cosine(embeddings_mosi[i], embeddings_iemocap[j])
            cosine_sims_mosi2iemocap[i][j] = cosine_sim_i_j

    print('calculate the simility of meld to mosi & mosei')
    for i in tqdm(range(meld_len)):
        for j in range(mosi_len):
            cosine_sim_i_j = 1 - cosine(embeddings_meld[i], embeddings_mosi[j])
            cosine_sims_meld2mosi[i][j] = cosine_sim_i_j

    print('calculate the simility of iemocap to mosi & mosei')
    for i in tqdm(range(iemocap_len)):
        for j in range(mosi_len):
            cosine_sim_i_j = 1 - cosine(embeddings_iemocap[i], embeddings_mosi[j])
            cosine_sims_iemocap2mosi[i][j] = cosine_sim_i_j

    print('embeddings calculation finished!')

    mosi_labels = {}
    meld_labels = {}
    iemocap_labels = {}

    for i in range(mosi_len):
        meld_ix_row = np.argmax(cosine_sims_mosi2meld[i,:])
        iemocap_ix_row = np.argmax(cosine_sims_mosi2iemocap[i,:])
        
        if (cosine_sims_mosi2meld[i, meld_ix_row] > cosine_sims_mosi2iemocap[i, iemocap_ix_row]):
            label = (meld_sen[meld_ix_row][3], 'meld')
        else:
            label = (iemocap_sen[iemocap_ix_row][2], 'iemocap')
        
        id = mosi_sen[i][0]
        if id not in mosi_labels.keys():
            mosi_labels[id] = label
        else:
            print('{} already exists'.format(str(id)))

    for i in range(meld_len):
        mosi_ix_row = np.argmax(cosine_sims_meld2mosi[i,:])

        id = meld_sen[i][0]
        label = mosi_sen[mosi_ix_row][3]
        if id not in meld_labels.keys():
            if mosi_ix_row < mosi_len:
                meld_labels[id] = (label, 'mosi')
            else:
                meld_labels[id] = (label, 'mosei')
        else:
            print('{} already exists'.format(str(id)))

    for i in range(iemocap_len):
        mosi_ix_row = np.argmax(cosine_sims_iemocap2mosi[i,:])

        id = iemocap_sen[i][0]
        label = mosi_sen[mosi_ix_row][3]
        if id not in iemocap_labels.keys():
            if mosi_ix_row < mosi_len:
                iemocap_labels[id] = (label, 'mosi')
            else:
                iemocap_labels[id] = (label, 'mosei')
        else:
            print('{} already exists'.format(str(id)))

    return mosi_labels, meld_labels, iemocap_labels

def meld_generate(path, path_out, pos_id2label, neg_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        id = str(line['Dialogue_ID']) + '_' + str(line['Utterance_ID'])
        if id in pos_id2label.keys():
            label, cross_dataset = pos_id2label[id]
        elif id in neg_id2label.keys():
            label, cross_dataset = neg_id2label[id]
        else:
            label, cross_dataset = 0.0, 'neutral'
        data_out.append({'Dialogue_ID': line['Dialogue_ID'], 'Utterance_ID': line['Utterance_ID'],
                         'Season': line['Season'], 'Episode': line['Episode'],
                         'Speaker': line['Speaker'], 'Utterance': line['Utterance'],
                         'Emotion': line['Emotion'], 'Sentiment': line['Sentiment'],
                         'score_label': label, 'cross_dataset': cross_dataset})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=0)

def iemocap_generate(path, path_out, pos_id2label, neg_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        id = line['vid_cid']
        label = line['label']
        por = 'None'
        if label == 'neu':
            por = 'neutral'
        elif label in ['hap', 'exc']:
            por = 'positive'
        else:
            por = 'negative'
        if id in pos_id2label.keys():
            label, cross_dataset = pos_id2label[id]
        elif id in neg_id2label.keys():
            label, cross_dataset = neg_id2label[id]
        else:
            label, cross_dataset = 0.0, 'neutral'
        data_out.append({'vid_cid': line['vid_cid'], 'label': line['label'],
                         'VAD': line['VAD'], 'text': line['text'], 'vid': line['vid'],
                         'por': por, 'score_label': label, 'cross_dataset': cross_dataset})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=0)

def mosi_generate(path, path_out, pos_id2label, neg_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        vid = line['video_id']
        clip_id = line['clip_id']
        id = str(vid) + '_' + str(clip_id)
        if id in pos_id2label.keys():
            label, cross_dataset = pos_id2label[id]
        elif id in neg_id2label.keys():
            label, cross_dataset = neg_id2label[id]
        else:
            label, cross_dataset = 'neutral', 'neutral'
        data_out.append({'video_id': vid, 'clip_id': clip_id, 'text': line['text'],
                         'score_label': line['label'], 'annotation': line['annotation'],
                         'cross_label': label, 'cross_dataset': cross_dataset,
                         'mode': line['mode'], 'label_by': line['label_by']})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=0)


name = 'princeton-nlp/sup-simcse-bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(DEVICE)
model.eval()

mosi_path = '/content/MOSI-label.csv'
mosei_path = '/content/MOSEI-label.csv'
meld_path = '/content/all_sent_emo.csv'
iemocap_path = '/content/IEMOCAP_all.csv'

(meld_pos, meld_neu, meld_neg), (mosi_pos, mosi_neu, mosi_neg), (iemocap_pos, iemocap_neu, iemocap_neg) = polarity_set_2v(mosi_path, mosei_path, iemocap_path, meld_path)

meld_train_path = '/content/train_sent_emo.csv'
meld_dev_path = '/content/dev_sent_emo.csv'
meld_test_path = '/content/test_sent_emo.csv'

meld_train_out_path = '/content/new_train_sent_emo-v3.csv'
meld_dev_out_path = '/content/new_dev_sent_emo-v3.csv'
meld_test_out_path = '/content/new_test_sent_emo-v3.csv'

mosi_out_path = '/content/new_MOSI-label-v3.csv'
mosei_out_path = '/content/new_MOSEI-label-v3.csv'
iemocap_out_path = '/content/new-iemocap-label-v3.csv'

mosi_pos_label, meld_pos_label, iemocap_pos_label = cal_cosine_sim_2v(mosi_pos, meld_pos, iemocap_pos, mosi_len_pos)

mosi_neg_label, meld_neg_label, iemocap_neg_label = cal_cosine_sim_2v(mosi_neg, meld_neg, iemocap_neg, mosi_len_neg)

mosi_generate(mosi_path, mosi_out_path, mosi_pos_label, mosi_neg_label)
mosi_generate(mosei_path, mosei_out_path, mosi_pos_label, mosi_neg_label)

meld_generate(meld_train_path, meld_train_out_path, meld_pos_label, meld_neg_label)
meld_generate(meld_dev_path, meld_dev_out_path, meld_pos_label, meld_neg_label)
meld_generate(meld_test_path, meld_test_out_path, meld_pos_label, meld_neg_label)

iemocap_generate(iemocap_path, iemocap_out_path, iemocap_pos_label, iemocap_neg_label)