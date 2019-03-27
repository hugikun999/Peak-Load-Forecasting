import argparse
import numpy as np
import mxnet as mx
from Net import lstm, Net
from electric_dataset_pd import electric_dataset

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Preday', default=60, type=int,
                      help='how many days used to train model')
    parser.add_argument('--FeatureList', default=[1, -3], type=list,
                      help='which feature used to train model')
    parser.add_argument('--IgnoreNorList', default=[-3], type=list,
                        help='which feature needed to ignore normalization')
    parser.add_argument('--start_day', default='20190301', type=str,
                        help='Days after the day you want to predict')
    parser.add_argument('--gpus', default=0, type=int,
                        help='gpu id, -1 mean use cpu')
    parser.add_argument('--model', default='lstm', type=str,
                        help='Use which model, lstm or cnn')
    parser.add_argument('--weights', default='test.params', type=str)
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

if __name__ == '__main__':
    args = process_args()
    model, ctx = get_model_ctx(args)

    model.load_parameters(args.weights)
    model.collect_params().reset_ctx(ctx)

    dataset = electric_dataset('data.csv', args.Preday, feature=args.FeatureList, ignore_nor=args.IgnoreNorList, inference=True, inf_startday=args.start_day)
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