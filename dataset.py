'''
* @name: dataset.py
* @description: Dataset loading functions. Note: The code source references MMSA (https://github.com/thuiar/MMSA/tree/master).
'''


import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler

__all__ = ['MMDataLoader', 'get_IEMOCAP_loaders', 'get_MELD_loaders']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims
        }
        DATA_MAP[args.dataset]()

    def __init_mosi(self):
        path = self.args.data_path
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # self.args.use_bert = True
        # self.args.need_truncated = True
        # self.args.need_data_aligned = True

        # if self.args.use_bert:
        self.text = data[self.mode]['text_bert'].astype(np.float32)
        # else:
            # self.text = data[self.mode]['text'].astype(np.float32)
     
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        # self.labels = {
        #     'M': data[self.mode]['regression'+'_labels'].astype(np.float32)
        # }
        self.labels = {}
        if 'regression'+'_labels' in data[self.mode]:
            self.labels['M'] = data[self.mode]['regression'+'_labels'].astype(np.float32)
        else:
            self.labels['M'] = None

        if 'classification_labels' in data[self.mode]:
            self.labels['ER'] = data[self.mode]['classification_labels'].astype(np.float32)
        else:
            self.labels['ER'] = None

        if self.args.dataset == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode]['regression'+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # if not self.args.need_data_aligned:
            # self.audio_lengths = data[self.mode]['audio_lengths']
            # self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        # if self.args.need_truncated:
        #     self.__truncated()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        labels_dict = {}
        for k, v in self.labels.items():
            if v is None:
                labels_dict[k] = None
            else:
                labels_dict[k] = torch.Tensor(v[index].reshape(-1))

        sample = {
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'id': self.ids[index],
            'labels': labels_dict
        } 
        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                    #    num_workers=args.num_workers,
                       shuffle=True,
                       generator=torch.Generator(device='cuda'))
        for ds in datasets.keys()
    }
    
    return dataLoader


class IEMOCAPDataset(Dataset):
    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, self.testVid = pickle.load(
            open('datasets/IEMOCAP/iemocap_multi_features.pkl', 'rb'), encoding='latin1')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        # return torch.FloatTensor(self.videoText[vid]), torch.FloatTensor(
        #     self.videoVisual[vid]), torch.FloatTensor(self.videoAudio[vid]), torch.FloatTensor(
        #         [[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]]), torch.FloatTensor(
        #             [1]*len(self.videoLabels[vid])), torch.LongTensor(self.videoLabels[vid]), vid
        sentiment = None
        emotion = torch.LongTensor(self.videoLabels[vid])
        return {
            'text': torch.FloatTensor(self.videoText[vid]),
            'audio': torch.FloatTensor(self.videoAudio[vid]),
            'vision': torch.FloatTensor(self.videoVisual[vid]),
            'id': vid,
            'labels': {
                'M': sentiment,
                'ER': emotion
            }
        }

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True)
                if i<6 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, _, self.videoText, self.roberta2, self.roberta3, self.roberta4, self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        # return torch.FloatTensor(self.videoText[vid]), torch.FloatTensor(
        #     self.videoVisual[vid]), torch.FloatTensor(self.videoAudio[vid]), torch.FloatTensor(
        #         self.videoSpeakers[vid]), torch.FloatTensor(
        #             [1]*len(self.videoLabels[vid])), torch.LongTensor(self.videoLabels[vid]), vid
        sentiment = None
        emotion = torch.LongTensor(self.videoLabels[vid])
        return {
            'text': torch.FloatTensor(self.videoText[vid]),
            'audio': torch.FloatTensor(self.videoAudio[vid]),
            'vision': torch.FloatTensor(self.videoVisual[vid]),
            'id': vid,
            'labels': {
                'M': sentiment,
                'ER': emotion
            }
        }

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True)
                if i<6 else dat[i].tolist() for i in dat]

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_IEMOCAP_loaders(args):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn)
    valid_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             collate_fn=testset.collate_fn)
    return train_loader, valid_loader, test_loader

def get_MELD_loaders(args):
    trainset = MELDDataset('datasets/MELD/meld_multi_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn)
    valid_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn)

    testset = MELDDataset('datasets/MELD/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=args.batch_size,
                             collate_fn=testset.collate_fn)
    return train_loader, valid_loader, test_loader