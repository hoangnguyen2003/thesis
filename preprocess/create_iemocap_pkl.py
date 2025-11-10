import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pkl_path = "datasets/IEMOCAP/iemocap_multi_features.pkl"
csv_path = "datasets/csv/iemocap-label.csv"
out_path = "datasets/final/iemocap.pkl"

with open(pkl_path, "rb") as f:
    old_data = pickle.load(f)

videoIDs, _, _, videoText, _, _, _, videoAudio, videoVisual, videoSentence, trainVid, testVid = old_data

df = pd.read_csv(csv_path)

def build_split(vid_list):
    ids, sentences, texts, audios, visuals, cls_labels, reg_labels = [], [], [], [], [], [], []
    for key in vid_list:
        if key not in videoIDs:
            continue
        for i, vid_cid in enumerate(videoIDs[key]):
            if vid_cid not in df['vid_cid'].values:
                continue
            row = df[df['vid_cid'] == vid_cid].iloc[0]
            ids.append(vid_cid)
            sentences.append(videoSentence[key][i])
            texts.append(videoText[key][i])
            audios.append(videoAudio[key][i])
            visuals.append(videoVisual[key][i])
            cls_labels.append(row['emotion_id'])
            reg_labels.append(row['score_label'])
    return {
        'id': ids,
        'videoSentence': sentences,
        'videoText': texts,
        'videoAudio': audios,
        'videoVisual': visuals,
        'classification_labels': cls_labels,
        'regression_labels': reg_labels
    }

train_split = build_split(trainVid)
test_split = build_split(testVid)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_split['id'])), test_size=0.09, random_state=42
)

def subset_split(split, indices):
    return {k: [v[i] for i in indices] for k, v in split.items()}

valid_split = subset_split(train_split, valid_idx)
train_split = subset_split(train_split, train_idx)

new_data = {
    'train': train_split,
    'valid': valid_split,
    'test': test_split
}

with open(out_path, "wb") as f:
    pickle.dump(new_data, f)