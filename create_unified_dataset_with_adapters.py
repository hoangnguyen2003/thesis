import argparse
import random
import pickle
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

import dataset as dataset_module  

def flatten_conversation_sample(s):
    t_tensor = s['text']
    a_tensor = s['audio']
    v_tensor = s['vision']
    er = s['labels'].get('ER', None)
    L = min(t_tensor.shape[0], a_tensor.shape[0], v_tensor.shape[0])
    out = []
    for i in range(L):
        lab_er = None
        if isinstance(er, torch.Tensor) and er.numel() >= i+1:
            lab_er = er[i].unsqueeze(0)
        sample = {
            'text': t_tensor[i].unsqueeze(0).clone(),
            'audio': a_tensor[i].unsqueeze(0).clone(),
            'vision': v_tensor[i].unsqueeze(0).clone(),
            'id': f"{s.get('id','conv')}_{i}",
            'labels': {'M': None, 'ER': lab_er}
        }
        out.append(sample)
    return out

def make_utterance_list_from_dataset(ds, name):
    out = []
    for i in range(len(ds)):
        s = ds[i]
        er = s['labels'].get('ER', None)
        m = s['labels'].get('M', None)
        if isinstance(er, torch.Tensor) and er.numel() > 1:
            flats = flatten_conversation_sample(s)
            for x in flats:
                x['_source'] = name
            out.extend(flats)
        else:
            sample = {
                'text': s['text'].clone(),
                'audio': s['audio'].clone(),
                'vision': s['vision'].clone(),
                'id': s.get('id', f'{name}_{i}'),
                'labels': {'M': None, 'ER': None},
                '_source': name
            }
            if isinstance(m, torch.Tensor) and m.numel() == 1:
                sample['labels']['M'] = m.clone().reshape(1)
            if isinstance(er, torch.Tensor) and er.numel() == 1:
                sample['labels']['ER'] = er.clone().reshape(1).long()
            out.append(sample)
    return out

class MultimodalEncoder(nn.Module):
    def __init__(self, text_dim, audio_dim, vis_dim, hidden=512, embed_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.vis_proj = nn.Linear(vis_dim, hidden)
        self.fuse = nn.Sequential(nn.Linear(hidden*3, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))

    def forward(self, text_seq, audio_seq, vis_seq):
        # text_seq: (B, Tt, dtarget)
        t = text_seq.mean(dim=1)
        a = audio_seq.mean(dim=1)
        v = vis_seq.mean(dim=1)

        t = self.text_proj(t)
        a = self.audio_proj(a)
        v = self.vis_proj(v)

        concat = torch.cat([t, a, v], dim=-1)
        z = self.fuse(concat)

        return F.normalize(z, dim=-1)

class ProjHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def nt_xent_from_pairs(zA, zB, temp=0.07):
    N = zA.shape[0]
    z = torch.cat([zA, zB], dim=0)
    sim = (z @ z.T) / temp
    mask = torch.eye(2*N, device=sim.device).bool()
    sim.masked_fill_(mask, -1e12)
    labels = torch.arange(N, device=sim.device)
    pos_idx = torch.cat([labels + N, labels])
    return F.cross_entropy(sim, pos_idx)

def pad_time_and_stack(list_of_tensors):
    maxT = max(x.shape[0] for x in list_of_tensors)
    D = list_of_tensors[0].shape[1]
    out = []
    for x in list_of_tensors:
        if x.shape[0] < maxT:
            pad = torch.zeros((maxT - x.shape[0], D), dtype=x.dtype, device=x.device)
            out.append(torch.cat([x, pad], dim=0))
        else:
            out.append(x)
    return torch.stack(out, dim=0)

IEMO_ID_TO_NAME = {0: 'hap', 1: 'sad', 2: 'neu', 3: 'ang', 4: 'exc', 5: 'fru'}
MELD_ID_TO_NAME = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}
IEMO_NAME_TO_NORMALIZED = {
    'hap': 'joy',
    'sad': 'sadness',
    'neu': 'neutral',
    'ang': 'anger',
    'exc': 'exc',
    'fru': 'fru'
}
MELD_NAME_TO_NORMALIZED = {
    'neutral': 'neutral','surprise':'surprise','fear':'fear','sadness':'sadness','joy':'joy','disgust':'disgust','anger':'anger'
}
UNIFIED_NAMES = sorted(list({
    *IEMO_NAME_TO_NORMALIZED.values(),
    *MELD_NAME_TO_NORMALIZED.values()
}))
UNIFIED_NAME_TO_ID = {name: idx for idx, name in enumerate(UNIFIED_NAMES)}
UNIFIED_ID_TO_NAME = {v:k for k,v in UNIFIED_NAME_TO_ID.items()}
EMO_NAME_TO_POL = {'joy': 'pos', 'exc': 'pos', 'hap': 'pos',
                   'neutral': 'neu', 'neu': 'neu', 'surprise': 'neu',
                   'sadness': 'neg', 'sad': 'neg', 'anger': 'neg', 'ang': 'neg',
                   'disgust': 'neg', 'fear': 'neg', 'fru': 'neg'}


# helper to map (source, original_emotion_id) -> unified_id
def map_original_emotion_to_unified(src, orig_eid):
    if src.startswith('iemo') or src.startswith('iemocap'):
        name = IEMO_ID_TO_NAME.get(int(orig_eid), None)
        if name is None:
            return None
        norm = IEMO_NAME_TO_NORMALIZED.get(name, name)
    else:
        # assume MELD-like
        name = MELD_ID_TO_NAME.get(int(orig_eid), None)
        if name is None:
            return None
        norm = MELD_NAME_TO_NORMALIZED.get(name, name)
    uid = UNIFIED_NAME_TO_ID.get(norm, None)
    return uid  # could be None if mapping missing

def emo_unified_id_to_polarity(unified_id):
    name = UNIFIED_ID_TO_NAME.get(int(unified_id), None)
    if name is None:
        return 'neu'
    return EMO_NAME_TO_POL.get(name, 'neu')

def pipeline(args):
    device = torch.device(args.device)
    
    ds_col = []
    if args.mosi:
        mosi = dataset_module.MMDataset(argparse.Namespace(data_path=args.mosi, dataset='mosi'), mode='train')
        ds_col.append(('mosi', mosi))
    if args.mosei:
        mosei = dataset_module.MMDataset(argparse.Namespace(data_path=args.mosei, dataset='mosei'), mode='train')
        ds_col.append(('mosei', mosei))
    if args.meld:
        meld =dataset_module.MELDDataset(args.meld)
        ds_col.append(('meld', meld))
    if args.iemocap:
        iemo = dataset_module.IEMOCAPDataset()
        ds_col.append(('iemocap', iemo))

    # flatten and collect utterances
    utterances = []
    for name, ds in ds_col:
        ulist = make_utterance_list_from_dataset(ds, name)
        for u in ulist:
            u['_source'] = name
        utterances.extend(ulist)
    
    print('Loaded utterances:', len(utterances))

    # detect per-source dims
    src_dims = {}
    for u in utterances:
        src = u['_source']
        tdim = u['text'].shape[1]
        adim = u['audio'].shape[1]
        vdim = u['vision'].shape[1]
        if src not in src_dims:
            src_dims[src] = {'text': tdim, 'audio': adim, 'vision': vdim}

    print('Per-source dims detected:')
    for s, d in src_dims.items():
        print(' ', s, d)

    # choose target dims: default = max across sources
    target_text = args.target_text_dim if args.target_text_dim > 0 else max(d['text'] for d in src_dims.values())
    target_audio = args.target_audio_dim if args.target_audio_dim > 0 else max(d['audio'] for d in src_dims.values())
    target_vis = args.target_vis_dim if args.target_vis_dim > 0 else max(d['vision'] for d in src_dims.values())
    print('Target dims -> text:', target_text, 'audio:', target_audio, 'vision:', target_vis)

    # build adapters per source
    text_adapters = nn.ModuleDict()
    audio_adapters = nn.ModuleDict()
    vis_adapters = nn.ModuleDict()
    for s, d in src_dims.items():
        if d['text'] != target_text:
            text_adapters[s] = nn.Linear(d['text'], target_text)
        else:
            text_adapters[s] = nn.Identity()
            
        if d['audio'] != target_audio:
            audio_adapters[s] = nn.Linear(d['audio'], target_audio)
        else:
            audio_adapters[s] = nn.Identity()
            
        if d['vision'] != target_vis:
            vis_adapters[s] = nn.Linear(d['vision'], target_vis)
        else:
            vis_adapters[s] = nn.Identity()

    text_adapters = text_adapters.to(device); audio_adapters = audio_adapters.to(device); vis_adapters = vis_adapters.to(device)

    # build model that expects target dims
    encoder = MultimodalEncoder(target_text, target_audio, target_vis, hidden=args.hidden, embed_dim=args.embed_dim).to(device)
    head_msa = ProjHead(args.embed_dim, args.proj_dim).to(device)
    head_erc = ProjHead(args.embed_dim, args.proj_dim).to(device)

    # dataset loader: return list of samples per batch (no stacking yet)
    class SimpleListDataset:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]
    loader = DataLoader(SimpleListDataset(utterances), batch_size=args.batch_size,
                        shuffle=True, collate_fn=lambda x: x)

    optim = torch.optim.Adam(list(encoder.parameters())
                             + list(head_msa.parameters()) + list(head_erc.parameters())
                             + list(text_adapters.parameters())
                             + list(audio_adapters.parameters()) + list(vis_adapters.parameters()), lr=args.lr)

    # training loop: for each batch, apply adapters per-sample then pad/time-stack
    for epoch in range(1, args.epochs + 1):
        encoder.train(); head_msa.train(); head_erc.train()
        total_loss = 0.0; steps = 0
        for batch in loader:
            # apply adapters per-sample -> produce adapted tensors (T_i, D_target)
            adapted_text = []
            adapted_audio = []
            adapted_vis = []
            raw_M = []
            raw_ER = []
            sources = []
            for s in batch:
                src = s.get('_source','unknown')
                sources.append(src)
                t = s['text'].to(device)
                a = s['audio'].to(device)
                v = s['vision'].to(device)

                t_ad = text_adapters[src](t)
                a_ad = audio_adapters[src](a)
                v_ad = vis_adapters[src](v)

                adapted_text.append(t_ad)
                adapted_audio.append(a_ad)
                adapted_vis.append(v_ad)

                raw_M.append(s['labels'].get('M', None))
                raw_ER.append(s['labels'].get('ER', None))

            # pad and stack time dimension
            text_b = pad_time_and_stack(adapted_text)    # (B, Tmax, target_text)
            audio_b = pad_time_and_stack(adapted_audio)
            vis_b = pad_time_and_stack(adapted_vis)

            z = encoder(text_b, audio_b, vis_b)  # (B, D)

            # build polarity buckets to create positive pairs
            msa_pol = defaultdict(list); erc_pol = defaultdict(list)
            for i in range(len(batch)):
                m = raw_M[i]
                er = raw_ER[i]
                if isinstance(m, torch.Tensor) and m.numel() == 1:
                    v = float(m.item())
                    p = 'pos' if v > args.pos_th else ('neg' if v < args.neg_th else 'neu')
                    msa_pol[p].append(i)
                if isinstance(er, torch.Tensor) and er.numel() == 1:
                    unified_eid = map_original_emotion_to_unified(sources[i], int(er.item()))
                    p = emo_unified_id_to_polarity(unified_eid)
                    erc_pol[p].append(i)

            posA = []; posB = []
            for p in ['pos', 'neu', 'neg']:
                A = msa_pol.get(p, [])
                B = erc_pol.get(p, [])
                if len(A) == 0 or len(B) == 0: continue
                k = min(len(A), len(B))
                selA = random.sample(A, k)
                selB = random.sample(B, k)
                posA.extend(selA); posB.extend(selB)
            if len(posA) == 0:
                continue

            msa_z = head_msa(z[posA])
            erc_z = head_erc(z[posB])

            loss = nt_xent_from_pairs(msa_z, erc_z, temp=args.temp)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += float(loss.item()); steps += 1

        avg = total_loss / (steps + 1e-12) if steps > 0 else 0.0
        print(f'[Epoch {epoch}/{args.epochs}] loss={avg:.4f} steps={steps}')

    encoder.eval(); head_erc.eval(); head_msa.eval()
    ref_erc_emb = []; ref_erc_lab = []
    ref_msa_emb = []; ref_msa_lab = []
    with torch.no_grad():
        for s in utterances:
            src = s['_source']
            t = text_adapters[src](s['text'].to(device)).unsqueeze(0)   # (1, T, D)
            a = audio_adapters[src](s['audio'].to(device)).unsqueeze(0)
            v = vis_adapters[src](s['vision'].to(device)).unsqueeze(0)

            z = encoder(t, a, v)
            if isinstance(s['labels'].get('ER', None), torch.Tensor) and s['labels']['ER'].numel() == 1:
                unified_eid = map_original_emotion_to_unified(src, int(s['labels']['ER'].item()))
                e = head_erc(z).cpu().numpy()[0]
                ref_erc_emb.append(e)
                ref_erc_lab.append(int(unified_eid))
            if isinstance(s['labels'].get('M', None), torch.Tensor) and s['labels']['M'].numel() == 1:
                m = head_msa(z).cpu().numpy()[0]
                ref_msa_emb.append(m)
                ref_msa_lab.append(float(s['labels']['M'].item()))

    knn_erc = None; knn_msa = None
    if len(ref_erc_emb) > 0:
        knn_erc = NearestNeighbors(n_neighbors=min(args.k, len(ref_erc_emb)), metric='cosine').fit(np.stack(ref_erc_emb))
    if len(ref_msa_emb) > 0:
        knn_msa = NearestNeighbors(n_neighbors=min(args.k, len(ref_msa_emb)), metric='cosine').fit(np.stack(ref_msa_emb))

    # produce unified output (pseudo-label missing ones)
    unified = []
    with torch.no_grad():
        for s in utterances:
            src = s['_source']
            entry = {'text': s['text'], 'audio': s['audio'], 'vision': s['vision'],
                     'id': s['id'], 'labels': {'M': None, 'ER': None}, '_source': src}
            
            if isinstance(s['labels'].get('M', None), torch.Tensor) and s['labels']['M'].numel() == 1:
                entry['labels']['M'] = s['labels']['M'].clone()
            if isinstance(s['labels'].get('ER', None), torch.Tensor) and s['labels']['ER'].numel() == 1:
                ueid = map_original_emotion_to_unified(src, int(s['labels']['ER'].item()))
                entry['labels']['ER'] = torch.tensor([int(ueid)], dtype=torch.long)

            t = text_adapters[src](s['text'].to(device)).unsqueeze(0)
            a = audio_adapters[src](s['audio'].to(device)).unsqueeze(0)
            v = vis_adapters[src](s['vision'].to(device)).unsqueeze(0)
            z = encoder(t, a, v)

            if entry['labels']['ER'] is None and knn_erc is not None:
                q = head_erc(z).cpu().numpy()[0].reshape(1,-1)
                dists, idxs = knn_erc.kneighbors(q)
                conf = 1.0 - float(dists.mean())
                if conf >= args.transfer_conf:
                    votes = [ref_erc_lab[idx] for idx in idxs[0]]
                    lab = Counter(votes).most_common(1)[0][0]
                    entry['labels']['ER'] = torch.tensor([int(lab)], dtype=torch.long)
                    entry['labels']['_erc_conf'] = conf

            if entry['labels']['M'] is None and knn_msa is not None:
                q = head_msa(z).cpu().numpy()[0].reshape(1,-1)
                dists, idxs = knn_msa.kneighbors(q)
                conf = 1.0 - float(dists.mean())
                if conf >= args.transfer_conf:
                    vals = [ref_msa_lab[idx] for idx in idxs[0]]
                    pred = float(np.mean(vals))
                    entry['labels']['M'] = torch.tensor([pred], dtype=torch.float)
                    entry['labels']['_msa_conf'] = conf
            unified.append(entry)

    print('Unified size:', len(unified))
    with open(args.out, 'wb') as f:
        pickle.dump(unified, f)
    print('Saved unified to', args.out)
    return unified

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mosi', type=str, default='/kaggle/input/cmu-mosi/aligned_50.pkl')
    p.add_argument('--mosei', type=str, default='/kaggle/input/cmu-mosei/aligned_50.pkl')
    p.add_argument('--meld', type=str, default='/kaggle/input/melddd/meld_multi_features.pkl')
    p.add_argument('--iemocap', type=str, default='/kaggle/input/iemocap/iemocap_multi_features.pkl')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--proj_dim', type=int, default=128)
    p.add_argument('--temp', type=float, default=0.07)
    p.add_argument('--pos_th', type=float, default=0.5)
    p.add_argument('--neg_th', type=float, default=-0.5)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--transfer_conf', type=float, default=0.6)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', type=str, default='unified_dataset.pkl')
    p.add_argument('--target_text_dim', type=int, default=0, help='0 -> auto (max across sources)')
    p.add_argument('--target_audio_dim', type=int, default=0)
    p.add_argument('--target_vis_dim', type=int, default=0)
    p.add_argument('--checkpoint_prefix', type=str, default='align_ckpt')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    pipeline(args)