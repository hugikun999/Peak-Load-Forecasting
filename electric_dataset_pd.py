import random, os
import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import gluon

class electric_dataset(gluon.data.Dataset):
    def __init__(self, root, preday, ratio=5, feature=[2, 5, 6, -1], ignore_nor=[4], training=True,
                 randomlist=None, inference=False, inf_startday='20180101'):
        self.preday = preday
        self.ratio = ratio
        self.feature = feature
        self.training = training
        self.inference = inference
        self.randomlist = randomlist
        self.ignore_nor = ignore_nor
        self.inf_startday = inf_startday

        self.df = pd.read_csv(root).set_index(['日期'])
        self.get_nor_params()
        if self.inference == True:
            self.test_split()
        else:
            self.loads = self.normalize(self.df)
            self.train_val_split()

    def test_split(self):
        daycnt = self.df.loc[self.inf_startday:].shape[0]
        predaycnt = (daycnt + self.preday)
        predata = self.df.iloc[predaycnt*-1:]
        predata = self.normalize(predata)

        Data = []
        for i in range(daycnt // 7):
            x = predata[i * 7: i * 7 + self.preday]
            y = predata[i * 7 + self.preday:i * 7 + self.preday + 7]
            tmp = [np.array(x), np.array(y)]
            Data.append(tmp)
        self.rest = daycnt % 7
        x = predata[-7 - self.preday:-7]
        y = predata[-7:]
        tmp = [np.array(x), np.array(y)]
        Data.append(tmp)
        self.loads = Data

    def train_val_split(self):
        totalday = self.loads.shape[0]
        Data = []
        for i in range(self.preday, totalday - 7):
            x = self.loads[i - self.preday:i]
            y = self.loads[i:i + 7]
            tmp = [np.array(x), np.array(y)]
            Data.append(tmp)
        totalday = len(Data)

        if self.training == True:
            valday = totalday // self.ratio
            tmp = []
            for i in range(totalday-1, totalday-valday-1, -1):
                tmp.append(i)
            self.randomlist = np.sort(random.sample(tmp, valday))
            for i in self.randomlist[::-1]:
                del Data[i]
            self.loads = Data
        else:
            self.loads = [Data[i] for i in self.randomlist]

    def get_nor_params(self):
        maxV = self.df.max()
        minV = self.df.min()
        meanV = self.df.mean()
        maxvalue = []
        minvalue = []
        mean = []
        for feature in self.feature:
            maxvalue.append(maxV[feature])
            minvalue.append(minV[feature])
            mean.append(meanV[feature])
        self.maxvalue = np.array(maxvalue)
        self.minvalue = np.array(minvalue)
        self.mean = np.array(mean)

    def normalize(self, data):
        tmp = []
        for idx, feature in enumerate(self.feature):
            if feature in self.ignore_nor:
                load = data.iloc[:, feature]
            else:
                load = (data.iloc[:, feature] - self.mean[idx]) / (self.maxvalue[idx] - self.minvalue[idx])
            tmp.append(load)
        data = np.concatenate(tmp).reshape(-1, data.shape[0]).transpose()
        return data


    def __getitem__(self, item):
        data = self.loads[item]
        x = data[0].transpose()
        y = data[1][:, 0]
        return mx.nd.array(x).reshape(len(self.feature), self.preday), mx.nd.array(y).reshape(1, 7)

    def __len__(self):
        return len(self.loads)


if __name__ == '__main__':

    feature_list = [1, -3, -2, -1]
    ignore_nor = [-3, -1]

    inference = electric_dataset('data.csv', 60, feature=feature_list, ignore_nor=ignore_nor, ratio=4, inference=True, inf_startday='20190301')
    train = electric_dataset('data.csv', 60, feature=feature_list, ignore_nor=ignore_nor, ratio=4)
    val = electric_dataset('data.csv', 60, feature=feature_list, randomlist=train.randomlist, training=False)

    train_iter = gluon.data.DataLoader(train, 1, shuffle=True)
    val_iter = gluon.data.DataLoader(val, 1, shuffle=False)
    for idx, (data, label) in enumerate(train_iter):
        print(data)
        print(data.shape)
        print(label)
        print(label.shape)
        os._exit(0)


