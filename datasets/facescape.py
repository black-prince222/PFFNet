import os
import glob
import numpy as np
import pickle
import torch.utils.data as data

__all__ = ['FaceScapeDataSet']


def load_pkl(path):
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file)
    return data


class FaceScapeDataSet(data.Dataset):
    def __init__(self,
                 npoints,
                 root,
                 train=True,
                 full=True):
        self.root = root
        self.train = train
        self.npoints = npoints
        self.samples = self.make_train_val_dataset(full)
        self.cache = {}
        self.cache_size = 30000

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index in self.cache:
            points1, points2, feat1, feat2, flow = self.cache[index]

        else:
            points1, points2, feat1, feat2 = self.data_loader(self.samples[index])
            flow = points2 - points1
            if len(self.cache) < self.cache_size:
                self.cache[index] = (points1, points2, feat1, feat2, flow)

        if self.train:
            n1 = points1.shape[0]
            sample_idx1 = np.random.choice(n1, n1, replace=False)
            n2 = points2.shape[0]
            sample_idx2 = np.random.choice(n2, n2, replace=False)

            points1 = points1[sample_idx1, :]
            points2 = points2[sample_idx2, :]
            feat1 = feat1[sample_idx1, :]
            feat2 = feat2[sample_idx2, :]
            flow = flow[sample_idx1, :]

        points1_center = np.mean(points1, 0)
        points1 -= points1_center
        points2 -= points1_center

        mask = np.ones([points1.shape[0]]).astype('float32')

        return points1, points2, feat1, feat2, flow, mask

    def make_train_val_dataset(self, full):
        person_face = os.listdir(self.root)
        person_face = sorted(list(map(lambda x: x.rjust(5, '0'), person_face)))
        person_face = list(map(lambda x: str(int(x)), person_face))

        if full:
            train_person_face = person_face[:600]
            val_person_face = person_face[600:]
        else:
            train_person_face = person_face[:300]
            val_person_face = person_face[600:]

        if self.train:
            train_list = []
            for person_id in train_person_face:
                train_list += glob.glob(os.path.join(self.root, person_id, '*.pkl'))
            train_list = list(filter(lambda x: '1_neutral.pkl' not in x, train_list))
            return train_list
        else:
            val_list = []
            for person_id in val_person_face:
                val_list += glob.glob(os.path.join(self.root, person_id, '*.pkl'))
            val_list = list(filter(lambda x: '1_neutral.pkl' not in x, val_list))
            return val_list

    def data_loader(self, path):
        data2 = load_pkl(path)
        points2 = data2['vertices'].astype('float32')
        color2 = data2['color'].astype('float32')
        normals2 = data2['normals'].astype('float32')
        feature2 = np.hstack([color2 / 255., normals2])

        neutral_path = os.path.join(os.path.dirname(path), '1_neutral.pkl')
        data1 = load_pkl(neutral_path)
        points1 = data1['vertices'].astype('float32')
        color1 = data1['color'].astype('float32')
        normals1 = data1['normals'].astype('float32')
        feature1 = np.hstack([color1 / 255., normals1])

        return points1, points2, feature1, feature2
