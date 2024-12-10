from .base import BASEDataModule
from alm.data.A2F import A2FDataset
from transformers import Wav2Vec2Processor
from collections import defaultdict

import os
from os.path import join as pjoin
import pickle
from tqdm import tqdm
import librosa
import numpy as np
from multiprocessing import Pool

# # mean and std of (vertice-template) for each subject
# claire_mean = -0.0018
# claire_std = 0.0108
# mark_mean = -0.0007
# mark_std = 0.0094
# james_mean = -0.0019
# james_std = 0.0123

# not used for now
claire_mean = 0
claire_std = 1
mark_mean = 0
mark_std = 1
james_mean = 0
james_std = 1


def load_data(args):
    file, root_dir, processor, template, audio_dir, vertice_dir, id = args
    if file.endswith('wav'):
        wav_path = os.path.join(root_dir, audio_dir, file)
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        key = file.replace("wav", "npy")
        result = {}
        result["audio"] = input_values
        subject_id = id
        result["subject_id"] = subject_id
        result["name"] = file.replace(".wav", "")
        result["path"] = os.path.abspath(wav_path)

        # template is "subtracted" from the vertice, so the mean should be "added"
        if subject_id == 'claire':
            result["template"] = (template.reshape((-1)) + claire_mean) / claire_std
        elif subject_id == 'james':
            result["template"] = (template.reshape((-1)) + james_mean) / james_std    
        elif subject_id == 'mark':
            result["template"] = (template.reshape((-1)) + mark_mean) / mark_std
        else:
            raise ValueError(f"Unknown subject: {subject_id}")
        vertice_path = os.path.join(root_dir, vertice_dir, file.replace("wav", "npy"))
        if not os.path.exists(vertice_path):
            return None
        else:
            if subject_id == 'claire':  
                result["vertice"] = np.load(vertice_path,allow_pickle=True) / claire_std
            elif subject_id == 'james':
                result["vertice"] = np.load(vertice_path,allow_pickle=True) / james_std    
            elif subject_id == 'mark':
                result["vertice"] = np.load(vertice_path,allow_pickle=True) / mark_std
            else:
                raise ValueError(f"Unknown subject: {subject_id}")
            return (key, result)

class A2FDataModule(BASEDataModule):
    def __init__(self,
                cfg,
                batch_size,
                num_workers,
                collate_fn = None,
                phase="train",
                **kwargs):
        super().__init__(batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = 'A2F'
        self.Dataset = A2FDataset
        self.cfg = cfg
        
        # customized to VOCASET
        self.subjects = [
                'claire',
                'james',
                'mark',
        ]

        self.root_dir = kwargs.get('data_root', None)
        self.audio_dir = 'wav'
        self.vertice_dir = 'vertices_npy'
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.template_file = 'templates.pkl'

        self.nfeats = 72035

        # load
        data = defaultdict(dict)

        count = 0
        args_list = []
        for id in self.subjects:
            with open(os.path.join(self.root_dir, id, self.template_file), 'rb') as fin:
                template = pickle.load(fin, encoding='latin1')[id]
            for r, ds, fs in os.walk(os.path.join(self.root_dir, id, self.audio_dir)):
                for f in fs:
                    args_list.append((f, os.path.join(self.root_dir, id), processor, template, self.audio_dir, self.vertice_dir, id))

                # # comment off for full dataset
                # count += 1
                # if count > 10:
                #     break

        # split dataset
        self.data_splits = {
            'train':[],
            'val':[],
            'test':[],
        }

        motion_list = []

        if False: # multi-process
            with Pool(processes=os.cpu_count()) as pool:
                results = pool.map(load_data, args_list)
                for result in results:
                    if result is not None:
                        key, value = result
                        data[key] = value
        else: # single process
            for args in tqdm(args_list, desc="Loading data"):
                result = load_data(args)
                if result is not None:
                    key, value = result
                    data[key] = value
                else:
                    print("Warning: data not found")


        # # calculate mean and std
        # motion_list = np.concatenate(motion_list, axis=0)
        # self.mean = np.mean(motion_list, axis=0)
        # self.std = np.std(motion_list, axis=0)

        '''
        splits = {
                    'train':range(1,41),
                    'val':range(21,41),
                    'test':range(21,41)
                }
        '''
        val_tracks_claire = test_tracks_claire = ['cp18_neutral', 'cp32_anger']
        val_tracks_james = test_tracks_james = ['eg1_neutral', 'ep1_anger']
        val_tracks_mark = test_tracks_mark = ['p1_neutral', 'p1_anger']
        
        for k, v in data.items():
            #subject_id = "_".join(k.split("_")[:-1])
            #sentence_id = int(k.split(".")[0][-2:])
            subject_id = v['subject_id']
            if subject_id == 'claire':
                test_tracks = test_tracks_claire
                val_tracks = val_tracks_claire
            elif subject_id == 'james':
                test_tracks = test_tracks_james
                val_tracks = val_tracks_james
            elif subject_id == 'mark':
                test_tracks = test_tracks_mark
                val_tracks = val_tracks_mark
            else:
                raise ValueError(f"Unknown subject: {subject_id}")

            self.data_splits['train'].append(v) # use all data for training
            if k[:-4] in val_tracks:
                self.data_splits['val'].append(v)
            if k[:-4] in test_tracks:
                self.data_splits['test'].append(v)



    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        # question
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                # todo: config name not consistent
                self.__dict__[item_c] = self.Dataset(
                    data = self.data_splits[subset] ,
                    subjects = self.subjects,
                    data_type = subset
                )
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")