import pandas as pd

iemocap_map = {
    'hap': 0,
    'sad': 1,
    'neutral': 2,
    'ang': 3,
    'exc': 4,
    'fru': 5
}
meld_map = {
    'neutral': 0,
    'surprise': 1,
    'fear': 2,
    'sadness': 3,
    'joy': 4,
    'disgust': 5,
    'anger': 6
}

path = 'datasets/CMU-MOSEI/new_MOSEI-label-v3.csv'
out_path = 'datasets/CMU-MOSEI/MOSEI-label.csv'

df = pd.read_csv(path)
df['iemocap_id'] = df['iemocap_label'].map(iemocap_map)
df['meld_id'] = df['meld_label'].map(meld_map)
df.to_csv(out_path, index=False)