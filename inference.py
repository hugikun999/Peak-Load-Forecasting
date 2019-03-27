import numpy as np
import mxnet as mx
from Net import lstm
from electric_dataset_pd import electric_dataset

if __name__ == '__main__':
    ctx = mx.gpu()
    model = lstm()
    model.load_parameters('test.params')
    model.collect_params().reset_ctx(ctx)

    feature_list = [1, -3]
    ignore_nor = [-3]
    preday = 60
    dataset = electric_dataset('data.csv', preday, feature=feature_list, ignore_nor=ignore_nor, inference=True, inf_startday='20190301')
    data = mx.gluon.data.DataLoader(dataset, 1, shuffle=False)

    RMSE = 0
    maxvalue = dataset.maxvalue
    minvalue = dataset.minvalue
    mean = dataset.mean
    for batch, (x, y) in enumerate(data):
        x = x.as_in_context(ctx)
        y = y.asnumpy().reshape(-1)

        pred = model(x).asnumpy().reshape(-1)
        if batch + 1 == dataset.__len__():
            pred = pred[dataset.rest*-1:]
            y = y[dataset.rest*-1:]

        pred = (pred * (maxvalue[0] - minvalue[0])) + mean[0]
        y = (y * (maxvalue[0] - minvalue[0])) + mean[0]
        RMSE += np.sqrt(((pred - y) ** 2).mean())
    RMSE /= dataset.__len__()
    print('RMSE:\t{}\n'.format(RMSE))