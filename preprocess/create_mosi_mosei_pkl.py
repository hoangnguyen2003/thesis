import pickle
import pandas as pd

# pkl_path = "datasets/CMU-MOSI/aligned_50.pkl"
# csv_path = "datasets/CMU-MOSI/MOSI-label.csv"
# output_path = "datasets/CMU-MOSI/cmu_mosi.pkl"
pkl_path = "datasets/CMU-MOSEI/aligned_50.pkl"
csv_path = "datasets/CMU-MOSEI/MOSEI-label.csv"
output_path = "datasets/CMU-MOSEI/cmu_mosei.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

df = pd.read_csv(csv_path)

df["id"] = df["video_id"].astype(str) + "$_$" + df["clip_id"].astype(str)

iemocap_map = dict(zip(df["id"], df["iemocap_id"]))
meld_map = dict(zip(df["id"], df["meld_id"]))

for split in ['train', 'valid', 'test']:
    updated_count = 0
    iemocap_ids = []
    meld_ids = []
    for i, sample_id in enumerate(data[split]["id"]):
        if sample_id in iemocap_map:
            iemocap_ids.append(iemocap_map[sample_id])
            meld_ids.append(meld_map[sample_id])
            updated_count += 1
    
    data[split]["iemocap_id"] = iemocap_ids
    data[split]["meld_id"] = meld_ids

    print(f"{updated_count}/{len(data[split]['id'])}")

with open(output_path, "wb") as f:
    pickle.dump(data, f)