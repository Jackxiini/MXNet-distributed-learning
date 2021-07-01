import os

os.environ.update({
  "DMLC_ROLE": "worker",
  "DMLC_PS_ROOT_URI": "10.157.6.183",
  "DMLC_PS_ROOT_PORT": "8000",
  "DMLC_NUM_SERVER": "1",
  "DMLC_NUM_WORKER": "2",
  "PS_VERBOSE": "2"
})

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxboard import SummaryWriter
import logging

logging.getLogger().setLevel(logging.DEBUG)

# 定义网络（gluon api）
net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))

# 初始参数
batch_size = 100
epochs = 10
learning_rate = 0.1
momentum = 0.9

# 导入数据
def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32)/255
    return data, label
kv_store = mx.kv.create('dist_async')

train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./mnist', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./mnist', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False)

# 预测函数
def testAccuacy(ctx):
    metric = mx.metric.Accuracy()
    for data, label in val_data: # here use the global val data
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data) # here use the global net
        metric.update([label], [output])

    return metric.get()

# 训练函数
def train(epochs, ctx):
    # 初始化图
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    net.hybridize() # 注意此处是混合图，必须进行hybridize操作

    # 定义训练器
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': learning_rate, 'momentum': momentum})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # 收集参数用于在每一步打印梯度等信息
    params = net.collect_params()
    param_names = params.keys()

    # 定义mxboard的writer，设置写入频率2s间隔
    sw = SummaryWriter(logdir='./logs', flush_secs=2)

    # 定义总步数
    global_step = 0

    for epoch in range(epochs):
        # 开始重置数据迭代器
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            # 在必要的时候将数据拷贝到ctx（比如gpu环境）
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # 后向梯度下降
            with autograd.record():
                output = net(data)
                L = loss(output, label)

            # 记录交叉熵
            sw.add_scalar(tag='cross_entropy', value=L.mean().asscalar(), global_step=global_step)

            # 后向计算
            global_step += 1
            L.backward()

            # 训练器步进
            trainer.step(data.shape[0])
            metric.update([label], [output])

            # 记录第一批图片
            if i == 0:
                sw.add_image('minist_first_minibatch', data.reshape((batch_size, 1, 28, 28)), epoch)

        # 第一次训练，记录图结构
        if epoch == 0:
            sw.add_graph(net)

        # 记录每步的梯度
        grads = [i.grad() for i in net.collect_params().values()]
        assert len(grads) == len(param_names)
        for i, name in enumerate(param_names):
            sw.add_histogram(tag=name, values=grads[i], global_step=epoch, bins=1000)

        # 记录训练数据预测精度
        name, train_acc = metric.get()
        print('[Epoch %d] Training: %s=%f' % (epoch, name, train_acc))
        sw.add_scalar(tag='accuracy_curves', value=('train_acc', train_acc), global_step=epoch)

        # 记录测试数据预测精度
        name, val_acc = testAccuacy(ctx)
        print('[Epoch %d] Validation: %s=%f' % (epoch, name, val_acc))
        sw.add_scalar(tag='accuracy_curves', value=('valid_acc', val_acc), global_step=epoch)

    # 关闭mxboard的writer
    sw.close()

    # 导出已训练的图
    net.export("mynet", epoch)

# main
ctx = mx.cpu()
train(epochs, ctx)