import argparse, pickle
import pandas as pd
import numpy as np
import mxnet as mx
from Net import lstm, Net
from electric_dataset_pd import electric_dataset

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Preday', default=60, type=int,
                      help='how many days used to train model')
    parser.add_argument('--FeatureList', default=[1, -4], nargs='+', type=int,
                      help='which feature used to train model')
    parser.add_argument('--IgnoreNorList', default=[-4], nargs='+', type=int,
                        help='which feature needed to ignore normalization')
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

def normalize(data, args, nor_params):
    tmp = []
    for idx, feature in enumerate(args.FeatureList):
        if feature in args.IgnoreNorList:
            load = data.iloc[:, feature]
        else:
            load = (data.iloc[:, feature] - nor_params[2]) / (nor_params[0] - nor_params[1])
        tmp.append(load)
    data = np.concatenate(tmp).reshape(-1, data.shape[0]).transpose()
    return data

if __name__ == '__main__':
    args = process_args()
    model, ctx = get_model_ctx(args)

    model.load_parameters(args.weights)
    model.collect_params().reset_ctx(ctx)
    #TODO modify nor value
    nor_params = np.array([37351, 19672, 29378.2126])

    df = pd.read_csv('data.csv').set_index(['日期'])
    data = df.iloc[args.Preday * -1:]
    data = normalize(data, args, nor_params)
    x = mx.ndarray.array(data, ctx=ctx).transpose().expand_dims(0)

    pred = model(x).asnumpy().reshape(-1)
    pred = np.round((pred * (nor_params[0] - nor_params[1])) + nor_params[2]).astype('int32')
    print(pred)
    # Process the holiday predicts
    with open('LRM.pickle', 'rb') as f:
        LRmodel = pickle.load(f)
        x = np.array(pred[2:4]).reshape(-1, 1)
        pred_holiday = LRmodel.predict(x).reshape(-1)
        pred[2:4] = pred_holiday
    print(pred)
    Date = ['20190402', '20190403', '20190404', '20190405', '20190406', '20190407', '20190408']
    df = pd.DataFrame({'date': Date, 'peak_load(MW)': pred})
    df.to_csv('submission.csv', index=False)