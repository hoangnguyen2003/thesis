import pandas as pd

emotion_map = {'angry': 0,
            'disgust': 1,
            'excited': 2,
            'fear': 3,
            'frustrated': 4,
            'joy': 5,
            'neutral': 6,
            'sad': 7,
            'surprise': 8}

paths = ['new_dev_sent_emo-v3.csv', 'new_MOSEI-label-v3.csv', 'new_MOSI-label-v3.csv',
        'new_test_sent_emo-v3.csv', 'new_train_sent_emo-v3.csv', 'new-iemocap-label-v3.csv']

out_paths = ['dev_sent_emo.csv', 'MOSEI-label.csv', 'MOSI-label.csv',
             'test_sent_emo.csv', 'train_sent_emo.csv', 'iemocap-label.csv']

for i, path in enumerate(paths):
    if path == 'new_dev_sent_emo-v3.csv' or path == 'new_test_sent_emo-v3.csv' or path == 'new_train_sent_emo-v3.csv':
        z = 'Emotion'
    elif path == 'new_MOSEI-label-v3.csv' or path == 'new_MOSI-label-v3.csv':
        z = 'cross_label'
    else:
        z = 'label'

    df = pd.read_csv(path)
    df[z] = df[z].replace("ang", "angry")
    df[z] = df[z].replace("anger", "angry")
    df[z] = df[z].replace("dis", "disgust")
    df[z] = df[z].replace("exc", "excited")
    df[z] = df[z].replace("fea", "fear")
    df[z] = df[z].replace("fru", "frustrated")
    df[z] = df[z].replace("hap", "joy")
    df[z] = df[z].replace("sadness", "sad")
    df[z] = df[z].replace("sur", "surprise")
    df[z] = df[z].replace("neu", "neutral")
    df["emotion_id"] = df[z].map(emotion_map)
    df.to_csv(out_paths[i], index=False)