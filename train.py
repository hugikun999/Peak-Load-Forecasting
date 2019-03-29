import numpy as np
import os, argparse
import mxnet as mx
from mxnet import autograd
from Net import Net, lstm
from electric_dataset_pd import electric_dataset

def process_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PredayList', default=60, nargs='+', type=int,
                      help='how many days used to train model')
    parser.add_argument('--FeatureList', default=[1, -3], nargs='+', type=int,
                      help='which feature used to train model')
    parser.add_argument('--IgnoreNorList', default=[-3], nargs='+', type=int,
                        help='which feature needed to ignore normalization')
    parser.add_argument('--gpus', default=0, type=int,
                        help='gpu id, -1 mean use cpu')
    parser.add_argument('--model', default='lstm', type=str,
                        help='Use which model, lstm or cnn')
    parser.add_argument('--ParamsNameList', default=['predict.params'], nargs='+', type=str)
    parser.add_argument('--SaveEpoch', default=[15], nargs='+', type=int)
    parser.add_argument('--MaxEpoch', default=100, type=int)
    return parser.parse_args()

def get_model_ctx(args):
    if args.model == 'lstm':
        net = lstm()
    elif args.model == 'cnn':
        net = Net()
    else:
        raise('Use model not be provided: {}'.format(args.model))

    if args.gpus > -1:
        ctx = mx.gpu(args.gpus)
    else:
        ctx = mx.cpu()

    return net, ctx

if __name__== '__main__':
    args = process_parser()
    assert len(args.ParamsNameList) == len(args.PredayList), 'length of ParamsNameList and PredayList must be equal'

    for preday, savename in zip(args.PredayList, args.ParamsNameList):
        net, ctx = get_model_ctx(args)
        net.initialize()
        net.collect_params().reset_ctx(ctx)
        trainer = mx.gluon.Trainer(net.collect_params(), 'adam',
                                   {'learning_rate': 1e-3})
        L2loss = mx.gluon.loss.L2Loss()

        train_dataset = electric_dataset('data.csv', preday, feature=args.FeatureList, ignore_nor=args.IgnoreNorList, ratio=4)
        val_dataset = electric_dataset('data.csv', preday, feature=args.FeatureList, randomlist=train_dataset.randomlist, training=False)

        train_iter = mx.gluon.data.DataLoader(train_dataset, 1, shuffle=True)
        val_iter = mx.gluon.data.DataLoader(val_dataset, 1, shuffle=False)

        for epoch in range(args.MaxEpoch):
            if epoch+1 == 10:
                lr = trainer.learning_rate
                trainer.set_learning_rate(lr * 0.1)
            for idx, (x, y) in enumerate(train_iter):

                x = x.as_in_context(ctx)
                y = y.as_in_context(ctx).reshape(1, -1)
                with autograd.record():
                    pred = net(x)
                    loss = L2loss(pred, y)

                loss.backward()
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

            if epoch in args.SaveEpoch:
                tmp = savename[:-7] + '_{}'.format(epoch) + savename[-7:]
                net.save_parameters(tmp)

        net.save_parameters(savename)