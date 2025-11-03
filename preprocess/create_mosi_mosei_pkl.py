import pickle
import pandas as pd

# pkl_path = "datasets/CMU-MOSI/aligned_50.pkl"
# csv_path = "datasets/csv/MOSI-label.csv"
# output_path = "datasets/final/cmu-mosi.pkl"
pkl_path = "datasets/CMU-MOSEI/aligned_50.pkl"
csv_path = "datasets/csv/MOSEI-label.csv"
output_path = "datasets/final/cmu-mosei.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

df = pd.read_csv(csv_path)

df["id"] = df["video_id"].astype(str) + "$_$" + df["clip_id"].astype(str)

label_map = dict(zip(df["id"], df["emotion_id"]))

for key in ['train', 'valid', 'test']:
    ids = data[key]["id"]
    labels = data[key]["classification_labels"]

    updated_count = 0
    for i, sample_id in enumerate(ids):
        if sample_id in label_map:
            new_label = label_map[sample_id]
            labels[i] = new_label
            updated_count += 1

    print(f"{updated_count}/{len(ids)}")

with open(output_path, "wb") as f:
    pickle.dump(data, f)