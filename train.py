import numpy as np
import csv, os
import mxnet as mx
from mxnet import autograd
from Net import Net, lstm
from electric_dataset_pd import electric_dataset

preday_list = [60]
feature_list = [1, -3, -2, -1]
ignore_nor = [-3, -1]
ctx = mx.gpu()
L2loss = mx.gluon.loss.L2Loss()

for preday in preday_list:
    net = lstm()
    net.initialize()
    net.collect_params().reset_ctx(ctx)
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam',
                               {'learning_rate': 1e-3})

    train_dataset = electric_dataset('data.csv', preday, feature=feature_list, ignore_nor=ignore_nor, ratio=4)
    val_dataset = electric_dataset('data.csv', preday, feature=feature_list, randomlist=train_dataset.randomlist, training=False)

    train_iter = mx.gluon.data.DataLoader(train_dataset, 1, shuffle=True)
    val_iter = mx.gluon.data.DataLoader(val_dataset, 1, shuffle=False)

    for epoch in range(100):
        if epoch+1 == 10:
            lr = trainer.learning_rate
            trainer.set_learning_rate(lr * 0.1)
        for idx, (x, y) in enumerate(train_iter):

            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx).reshape(1, -1)
            with autograd.record():
                pred = net(x)
                loss = L2loss(pred, y)
            # Compute gradients
            loss.backward()
            # Update parameters
            trainer.step(x.shape[0])
            loss.wait_to_read()

            if (idx+1) % 200 == 0:
                RMSE = 0
                maxvalue = train_dataset.maxvalue
                minvalue = train_dataset.minvalue
                mean = train_dataset.mean
                for (x, y) in val_iter:
                    x = x.as_in_context(ctx)
                    y = y.asnumpy().reshape(-1, 7)


                    pred = net(x).asnumpy().reshape(-1)
                    pred = (pred * (maxvalue[0] - minvalue[0])) + mean[0]
                    y = (y * (maxvalue[0] - minvalue[0])) + mean[0]

                    RMSE += np.sqrt(((pred-y)**2).mean())
                RMSE /= val_dataset.__len__()
                print('Preday[{}]Epoch[{}]Iter[{}]\tRMSE:\t{}\n'.format(preday, epoch, idx, RMSE))

        if epoch == 15:
            net.save_parameters('predict_{}.params'.format(preday))
    net.save_parameters('predict_final.params'.format(preday))