import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

def conv_block(channel, kernel=3, stride=1, pad=1):
    blc = nn.HybridSequential()
    blc.add(nn.Conv1D(channel, kernel_size=kernel, strides=stride, padding=pad),
            nn.BatchNorm(),
            nn.LeakyReLU(0.1))
    return blc

class Net(gluon.HybridBlock):
    def __init__(self):
        super(Net, self).__init__()
        with self.name_scope():
            self.conv1 = conv_block(32, kernel=7)
            self.conv2 = conv_block(64, kernel=5)
            self.conv3 = conv_block(128, kernel=3)
            self.dense1 = nn.Dense(1024)
            self.dense2 = nn.Dense(7)


    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

class lstm(nn.HybridBlock):
    def __init__(self):
        super(lstm, self).__init__()
        self.model = nn.HybridSequential()
        with self.model.name_scope():
            self.model.add(
                           mx.gluon.rnn.LSTM(20),
                           nn.Dense(7))
    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.model(x)


if __name__=='__main__':
    net = Net()
    net.initialize()
    net.collect_params().reset_ctx(mx.gpu())

    a = mx.ndarray.ones((1, 1, 60), ctx=mx.gpu())
    b = net(a).asnumpy()
    print(b.shape)
