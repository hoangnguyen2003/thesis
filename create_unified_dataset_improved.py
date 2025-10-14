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

def make_utterance_list_from_dataset(ds, name, map_fn):
    out = []
    for i in range(len(ds)):
        s = ds[i]
        er = s['labels'].get('ER', None)
        m = s['labels'].get('M', None)
        if isinstance(er, torch.Tensor) and er.numel() > 1:
            flats = flatten_conversation_sample(s)
            for x in flats:
                x['_source'] = name
                if isinstance(x['labels']['ER'], torch.Tensor) and x['labels']['ER'].numel() == 1:
                    u = map_fn(name, int(x['labels']['ER'].item()))
                    x['labels']['ER_unified'] = u
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
                sample['labels']['ER_unified'] = map_fn(name, int(er.item()))
            out.append(sample)
    return out

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
PREFERRED_ORDER = ['joy','neutral','anger','sadness','surprise','fear','disgust','exc','fru']
UNIFIED_NAMES = [n for n in PREFERRED_ORDER if n in set(list(IEMO_NAME_TO_NORMALIZED.values()) + list(MELD_NAME_TO_NORMALIZED.values()))]
UNIFIED_NAME_TO_ID = {name: idx for idx, name in enumerate(UNIFIED_NAMES)}
UNIFIED_ID_TO_NAME = {v:k for k,v in UNIFIED_NAME_TO_ID.items()}
EMO_NAME_TO_POL = {'joy': 'pos', 'exc': 'pos', 'hap': 'pos',
                   'neutral': 'neu', 'neu': 'neu', 'surprise': 'neu',
                   'sadness': 'neg', 'sad': 'neg', 'anger': 'neg', 'ang': 'neg',
                   'disgust': 'neg', 'fear': 'neg', 'fru': 'neg'}

# helper to map (source, original_emotion_id) -> unified_id
def map_original_emotion_to_unified(src, orig_eid):
    if str(src).lower().startswith('iemo') or str(src).lower().startswith('iemocap'):
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

def unified_id_to_polarity(unified_id):
    name = UNIFIED_ID_TO_NAME.get(int(unified_id), None)
    if name is None:
        return 'neu'
    return EMO_NAME_TO_POL.get(name, 'neu')

class AttentivePool(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.w = nn.Linear(inp_dim, 1)
    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.w(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (x * alpha).sum(dim=1)
        return pooled

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, dim, nhead=4, layers=1, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim*2, dropout=dropout, activation='relu')
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
    def forward(self, x):
        # x: (B, T, D) -> (T, B, D)
        out = self.tr(x.transpose(0,1)).transpose(0,1)
        return out

class PerModalityEncoder(nn.Module):
    def __init__(self, in_dim, target_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, target_dim)
        self.temporal = TemporalTransformerEncoder(target_dim, nhead=4, layers=1)
        self.pool = AttentivePool(target_dim)
    def forward(self, x):
        # x: (B, T, in_dim)
        h = self.proj(x)
        h = self.temporal(h)
        pooled = self.pool(h)
        return pooled, h

class ImprovedMultimodalModel(nn.Module):
    def __init__(self, text_dim, audio_dim, vis_dim, ttarget, atarget, vtarget, embed_dim=512, proj_dim=128, n_classes=8, proto_mom=0.9):
        super().__init__()
        # per-modality encoders expect adapted dims (target dims)
        self.text_enc = PerModalityEncoder(ttarget, ttarget)  # after adapter, dims == ttarget
        self.audio_enc = PerModalityEncoder(atarget, atarget)
        self.vis_enc = PerModalityEncoder(vtarget, vtarget)
        # fusion
        fusion_in = ttarget + atarget + vtarget
        self.fuse = nn.Sequential(nn.Linear(fusion_in, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        # heads
        self.proj_msa = nn.Sequential(nn.Linear(embed_dim, embed_dim//2), nn.ReLU(), nn.Linear(embed_dim//2, proj_dim))
        self.proj_erc = nn.Sequential(nn.Linear(embed_dim, embed_dim//2), nn.ReLU(), nn.Linear(embed_dim//2, proj_dim))
        self.clf_erc = nn.Linear(embed_dim, n_classes)
        self.reg_msa = nn.Linear(embed_dim, 1)
        # prototype buffer (proj space)
        self.register_buffer('prototypes', torch.zeros(n_classes, proj_dim))
        self.proto_momentum = proto_mom
    def forward(self, text, audio, vis):
        pt, _ = self.text_enc(text)
        pa, _ = self.audio_enc(audio)
        pv, _ = self.vis_enc(vis)
        fuse = torch.cat([pt, pa, pv], dim=-1)
        z = self.fuse(fuse)  # embed
        p_msa = F.normalize(self.proj_msa(z), dim=-1)
        p_erc = F.normalize(self.proj_erc(z), dim=-1)
        logits_erc = self.clf_erc(z)
        pred_m = self.reg_msa(z).squeeze(-1)
        return {'embed': z, 'proj_msa': p_msa, 'proj_erc': p_erc, 'logits_erc': logits_erc, 'pred_m': pred_m}
    def update_prototypes(self, proj_erc_batch, unified_labels_batch):
        # unified_labels_batch: tensor (n,) with -1 for unlabeled
        mask = unified_labels_batch >= 0
        if mask.sum()==0:
            return
        for c in torch.unique(unified_labels_batch[mask]):
            c = int(c.item())
            mean_vec = proj_erc_batch[unified_labels_batch==c].mean(dim=0)
            # momentum update
            self.prototypes[c] = self.proto_momentum * self.prototypes[c] + (1.0 - self.proto_momentum) * mean_vec.detach()

def nt_xent_pairs(zA, zB, temp=0.07):
    N = zA.shape[0]
    z = torch.cat([zA, zB], dim=0)
    sim = (z @ z.T) / temp
    mask = torch.eye(2*N, device=sim.device).bool()
    sim.masked_fill_(mask, -1e12)
    labels = torch.arange(N, device=sim.device)
    pos_idx = torch.cat([labels + N, labels])
    return F.cross_entropy(sim, pos_idx)

def prototype_loss(proj, unified_labels, prototypes, margin=0.2):
    mask = unified_labels >= 0
    if mask.sum() == 0:
        return proj.new_tensor(0.0)
    p = proj[mask]
    labs = unified_labels[mask]
    pos_proto = prototypes[labs]       # (n, D)
    sim_pos = (p * pos_proto).sum(-1)
    sim_all = p @ prototypes.t()
    sim_all[range(sim_all.shape[0]), labs] = -9e9
    sim_neg_max, _ = sim_all.max(dim=1)
    loss = F.relu(margin - sim_pos + sim_neg_max).mean()
    return loss

def pad_time_and_stack(list_of_tensors, device):
    maxT = max(x.shape[0] for x in list_of_tensors)
    D = list_of_tensors[0].shape[1]
    out = []
    for x in list_of_tensors:
        if x.shape[0] < maxT:
            pad = torch.zeros((maxT - x.shape[0], D), dtype=x.dtype, device=device)
            out.append(torch.cat([x.to(device), pad], dim=0))
        else:
            out.append(x.to(device))
    return torch.stack(out, dim=0)

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
        ulist = make_utterance_list_from_dataset(ds, name, map_original_emotion_to_unified)
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
    n_classes = len(UNIFIED_NAMES)
    model = ImprovedMultimodalModel(text_dim=None, audio_dim=None, vis_dim=None,
                                    ttarget=target_text, atarget=target_audio, vtarget=target_vis,
                                    embed_dim=args.embed_dim, proj_dim=args.proj_dim, n_classes=n_classes, proto_mom=args.proto_mom).to(device)
    params = list(model.parameters()) + list(text_adapters.parameters()) + list(audio_adapters.parameters()) + list(vis_adapters.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # dataset loader: return list of samples per batch (no stacking yet)
    class SimpleListDataset:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, idx): return self.items[idx]
    loader = DataLoader(SimpleListDataset(utterances), batch_size=args.batch_size,
                        shuffle=True, collate_fn=lambda x: x)

    # training loop: for each batch, apply adapters per-sample then pad/time-stack
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0; steps = 0
        for batch in loader:
            # batch is list of samples
            adapted_text=[]; adapted_audio=[]; adapted_vis=[]
            msa_label_list = []; er_unified_list = []
            # build lists
            for s in batch:
                src = s['_source']
                t = text_adapters[src](s['text'].to(device))
                a = audio_adapters[src](s['audio'].to(device))
                v = vis_adapters[src](s['vision'].to(device))
                adapted_text.append(t); adapted_audio.append(a); adapted_vis.append(v)
                # M label
                m = s['labels'].get('M', None)
                msa_label_list.append(m if isinstance(m, torch.Tensor) and m.numel()==1 else None)
                # unified ER if available (may have been set earlier)
                uer = s['labels'].get('ER_unified', None)
                # if original ER scalar present but not normalized field, map it
                if uer is None and isinstance(s['labels'].get('ER', None), torch.Tensor) and s['labels']['ER'].numel()==1:
                    uer = map_original_emotion_to_unified(s['_source'], int(s['labels']['ER'].item()))
                er_unified_list.append(uer if uer is not None else -1)
            # pad
            text_b = pad_time_and_stack(adapted_text, device)
            audio_b = pad_time_and_stack(adapted_audio, device)
            vis_b = pad_time_and_stack(adapted_vis, device)
            # forward
            out = model(text_b, audio_b, vis_b)
            proj_msa = out['proj_msa']; proj_erc = out['proj_erc']
            logits_erc = out['logits_erc']; pred_m = out['pred_m']

            # Build positive pairs for contrastive using polarity buckets
            # collect indices for msa-labeled and erc-labeled
            msa_idxs = []
            erc_idxs = []
            msa_pol = defaultdict(list); erc_pol = defaultdict(list)
            for i in range(len(batch)):
                if msa_label_list[i] is not None:
                    val = float(msa_label_list[i].item())
                    p = 'pos' if val > args.pos_th else ('neg' if val < args.neg_th else 'neu')
                    msa_pol[p].append(i); msa_idxs.append(i)
                if er_unified_list[i] is not None and er_unified_list[i] != -1:
                    uid = int(er_unified_list[i])
                    p = unified_id_to_polarity(uid)
                    erc_pol[p].append(i); erc_idxs.append(i)

            posA=[]; posB=[]
            for p in ['pos','neu','neg']:
                A = msa_pol.get(p, []); B = erc_pol.get(p, [])
                if len(A)==0 or len(B)==0: continue
                k = min(len(A), len(B))
                selA = random.sample(A, k); selB = random.sample(B, k)
                posA.extend(selA); posB.extend(selB)
                
            loss_nt = torch.tensor(0.0, device=device)
            if len(posA) > 0:
                loss_nt = nt_xent_pairs(proj_msa[posA], proj_erc[posB], temp=args.temp)
            # supervised ER CE
            er_labels_tensor = torch.tensor([er_unified_list[i] for i in range(len(batch))], device=device, dtype=torch.long)
            mask_er = er_labels_tensor >= 0
            loss_ce = torch.tensor(0.0, device=device)
            if mask_er.sum() > 0:
                loss_ce = F.cross_entropy(logits_erc[mask_er], er_labels_tensor[mask_er])
            # regression loss for M (MAE)
            msa_vals = torch.tensor([msa_label_list[i].item() if msa_label_list[i] is not None else float('nan') for i in range(len(batch))], device=device, dtype=torch.float)
            mask_m = ~torch.isnan(msa_vals)
            loss_m = torch.tensor(0.0, device=device)
            if mask_m.sum() > 0:
                loss_m = F.l1_loss(pred_m[mask_m], msa_vals[mask_m])
            # prototype loss
            loss_proto = prototype_loss(proj_erc, er_labels_tensor, model.prototypes, margin=args.proto_margin)
            # total loss
            total = args.alpha * loss_nt + args.beta * loss_ce + args.gamma * loss_proto + args.delta * loss_m
            optimizer.zero_grad(); total.backward(); optimizer.step()
            # update prototypes (momentum) using proj_erc and er_labels_tensor
            model.update_prototypes(proj_erc.detach(), er_labels_tensor.detach())
            total_loss += float(total.item()); steps += 1
        avg = total_loss / (steps+1e-12) if steps>0 else 0.0
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg:.4f} steps={steps}")
        # optional checkpoint saving
        if epoch % args.save_every == 0:
            ckpt = {'model': model.state_dict(), 'text_adapters': text_adapters.state_dict(),
                    'audio_adapters': audio_adapters.state_dict(), 'vis_adapters': vis_adapters.state_dict()}
            torch.save(ckpt, f"{args.checkpoint_prefix}_ep{epoch}.pt")

    ref_erc_emb = []; ref_erc_lab = []
    ref_msa_emb = []; ref_msa_lab = []
    model.eval()
    with torch.no_grad():
        for s in utterances:
            src = s['_source']
            t = text_adapters[src](s['text'].to(device)).unsqueeze(0)
            a = audio_adapters[src](s['audio'].to(device)).unsqueeze(0)
            v = vis_adapters[src](s['vision'].to(device)).unsqueeze(0)
            out = model(t, a, v)
            p_erc = out['proj_erc'].cpu().numpy()[0]
            p_msa = out['proj_msa'].cpu().numpy()[0]
            # unified ER label if exists
            uer = s['labels'].get('ER_unified', None)
            if uer is None and isinstance(s['labels'].get('ER', None), torch.Tensor) and s['labels']['ER'].numel()==1:
                uer = map_original_emotion_to_unified(s['_source'], int(s['labels']['ER'].item()))
            if uer is not None and uer != -1:
                ref_erc_emb.append(p_erc); ref_erc_lab.append(int(uer))
            if isinstance(s['labels'].get('M', None), torch.Tensor) and s['labels']['M'].numel()==1:
                ref_msa_emb.append(p_msa); ref_msa_lab.append(float(s['labels']['M'].item()))
    knn_erc = None
    knn_msa = None
    if len(ref_erc_emb) > 0:
        knn_erc = NearestNeighbors(n_neighbors=min(args.k, len(ref_erc_emb)), metric='cosine').fit(np.stack(ref_erc_emb))
    if len(ref_msa_emb) > 0:
        knn_msa = NearestNeighbors(n_neighbors=min(args.k, len(ref_msa_emb)), metric='cosine').fit(np.stack(ref_msa_emb))

    unified_output = []
    summary_rows = []
    with torch.no_grad():
        for s in utterances:
            src = s['_source']
            entry = {'id': s['id'], 'source': src, 'labels': {'M_orig': None, 'ER_orig': None}, 'pseudo': {}}
            if isinstance(s['labels'].get('M', None), torch.Tensor) and s['labels']['M'].numel()==1:
                entry['labels']['M_orig'] = float(s['labels']['M'].item())
            if isinstance(s['labels'].get('ER', None), torch.Tensor) and s['labels']['ER'].numel()==1:
                entry['labels']['ER_orig'] = int(s['labels']['ER'].item())
            # forward
            t = text_adapters[src](s['text'].to(device)).unsqueeze(0)
            a = audio_adapters[src](s['audio'].to(device)).unsqueeze(0)
            v = vis_adapters[src](s['vision'].to(device)).unsqueeze(0)
            out = model(t,a,v)
            logits = out['logits_erc'][0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            clf_top1 = int(probs.argmax()); clf_conf = float(probs.max())
            # prototype assignment (cosine sim)
            proto = model.prototypes.cpu().numpy()  # (C, D)
            q = out['proj_erc'].cpu().numpy()[0].reshape(1,-1)  # (1,D)
            cos_sim = (proto @ q.T).squeeze()  # (C,)
            proto_top1 = int(cos_sim.argmax()); proto_conf = float(cos_sim.max())
            # kNN (if available)
            knn_top1 = None; knn_conf = 0.0
            if knn_erc is not None:
                dists, idxs = knn_erc.kneighbors(q)
                # weighted votes by exp(-dist/temp)
                weights = np.exp(-dists[0] / args.knn_temp)
                labs = [ref_erc_lab[i] for i in idxs[0]]
                agg = {}
                for l,w in zip(labs, weights): agg[l] = agg.get(l,0.0)+w
                knn_top1 = max(agg.items(), key=lambda x: x[1])[0]
                knn_conf = float(max(agg.values()) / sum(weights))
            # ensemble rule: require at least clf_top1 == proto_top1 and confidence average >= threshold
            conf_avg = (clf_conf + (proto_conf if proto_conf is not None else 0.0) + (knn_conf if knn_top1 is not None else 0.0)) / (2.0 + (1.0 if knn_top1 is not None else 0.0))
            assigned_er = None
            assigned_src = None
            if clf_top1 == proto_top1 and conf_avg >= args.transfer_conf:
                assigned_er = int(clf_top1); assigned_src = 'clf+proto'
            elif knn_top1 is not None and clf_top1 == knn_top1 and conf_avg >= args.transfer_conf:
                assigned_er = int(clf_top1); assigned_src = 'clf+knn'
            elif knn_top1 is not None and proto_top1 == knn_top1 and conf_avg >= args.transfer_conf:
                assigned_er = int(proto_top1); assigned_src = 'proto+knn'
            else:
                assigned_er = None
            # M pseudo via knn_msa or regression pred
            assigned_m = None; assigned_m_conf = 0.0
            if knn_msa is not None:
                dists_m, idxs_m = knn_msa.kneighbors(out['proj_msa'].cpu().numpy().reshape(1,-1))
                weights_m = np.exp(-dists_m[0] / args.knn_temp)
                vals = [ref_msa_lab[i] for i in idxs_m[0]]
                assigned_m = float(np.average(vals, weights=weights_m))
                assigned_m_conf = float(1.0 - dists_m.mean())
            else:
                # fallback to model regression prediction
                assigned_m = float(out['pred_m'].cpu().numpy()[0])
                assigned_m_conf = float(np.max(F.softmax(out['logits_erc'], dim=-1).cpu().numpy()))  # poor proxy
            # record into entry
            if assigned_er is not None:
                entry['pseudo']['ER_unified_id'] = assigned_er
                entry['pseudo']['ER_unified_name'] = UNIFIED_ID_TO_NAME[assigned_er]
                entry['pseudo']['ER_method'] = assigned_src
                entry['pseudo']['ER_conf'] = conf_avg
            if assigned_m is not None:
                entry['pseudo']['M'] = assigned_m
                entry['pseudo']['M_conf'] = assigned_m_conf
            unified_output.append({'text': s['text'], 'audio': s['audio'], 'vision': s['vision'], 'id': s['id'], 'labels': {'M': s['labels'].get('M', None), 'ER': s['labels'].get('ER', None)}, 'pseudo': entry['pseudo'], '_source': s['_source']})
            # summary row for CSV
            summary_rows.append({
                'id': entry['id'],
                'source': entry['source'],
                'ER_orig': entry['labels']['ER_orig'],
                'ER_orig_name': (UNIFIED_ID_TO_NAME.get(map_original_emotion_to_unified(entry['source'], entry['labels']['ER_orig'])) if entry['labels']['ER_orig'] is not None else None),
                'ER_pseudo_id': entry['pseudo'].get('ER_unified_id', None),
                'ER_pseudo_name': entry['pseudo'].get('ER_unified_name', None),
                'ER_method': entry['pseudo'].get('ER_method', None),
                'ER_conf': entry['pseudo'].get('ER_conf', None),
                'M_orig': entry['labels']['M_orig'],
                'M_pseudo': entry['pseudo'].get('M', None),
                'M_conf': entry['pseudo'].get('M_conf', None)
            })

    # save outputs
    out_dict = {
        'unified': unified_output,
        'unified_name_to_id': UNIFIED_NAME_TO_ID,
        'unified_id_to_name': UNIFIED_ID_TO_NAME,
        'model_info': {'proj_dim': args.proj_dim, 'embed_dim': args.embed_dim}
    }
    with open(args.out, 'wb') as f:
        pickle.dump(out_dict, f)
    print("Saved unified pkl to", args.out)

    # write CSV summary
    csv_out = args.csv_out if args.csv_out else os.path.splitext(args.out)[0] + '_summary.csv'
    with open(csv_out, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=['id','source','ER_orig','ER_orig_name','ER_pseudo_id','ER_pseudo_name','ER_method','ER_conf','M_orig','M_pseudo','M_conf'])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print("Saved CSV summary to", csv_out)
    return out_dict

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