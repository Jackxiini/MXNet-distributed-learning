# MXNet分布式训练

## 并行类型
#### 1. 数据并行
每个设备存储一份完整的模型，通过对数据集不同部分训练来更新共享的模型。
*可以是单机多卡，也可以是多个机器*
#### 2. 模型并行
如果模型太大没办法放进设备的内存就可以使用这种方法。每个设备存储并训练学习模型的不同部分。

PS: 对网络带宽要求高，因为中间结果需要在不同节点之间传递，这会占用大量带宽而拉低整体计算效率。

*目前，MXNet对于模型并行这种方法只支持单机多卡。*

## 分布式训练如何工作
以下是MXNet分布式训练的原理知识
### 进程类型
#### 1. Worker
在一批训练样本上训练，在处理每个批次之前，Worker从Server上拉出权重。在每个批次处理后，Worker会向Server发送梯度。

PS：若训练模型工作量大，请不要在一台机器上运行多个Worker。
#### 2. Server
可有多个Servers，Server存储模型参数并与Workers交流。Server的进程可与Worker在一处，也可以不在一处。
#### 3. Scheduler
只能有一个。作用是配置集群。这包括等待每个节点启动以及节点正在监听哪个端口之类的消息。 然后Scheduler让所有进程知道集群中的其他节点的信息，以便它们可以相互通信。

### KVStore机制
Key-value存储机制，这是多设备训练中的关键部分。一个或多个Server通过将参数存储为key-value的形式在单台机器或者多台机器上进行跨节点的参数交互。这种存储机制中的每个值都由key-value表示，其中key是网络中参数数组，value是此参数数组的权重。Workers在一批计算后Push梯度，并且在新的批次计算开始之前Pull更新后的权重。我们也可以在更新每个权重时传入KVStore的优化器。这个优化器像随机梯度下降一样定义了一个更新规则——本质上是旧的权重、梯度和一些参数来计算新的权重。

如果你使用一个Gluon Trainer对象或者是模型的API，它将在内部使用KVStore来聚合梯度，这些梯度来自同一台机器上或者不同机器上的多个设备。

尽管无论是否使用多台机器进行训练，API都保持不变，但KVStore服务器的概念仅存在于分布式训练期间。在分布式情况下，每次push和pull都涉及与KVStore服务器的通信。当一台机器上有多个设备时，这些设备训练的梯度首先会聚合在机器上，然后发送再到服务器。

 PS: 我们需要构建标志`USE_DIST_KVSTORE = 1`之后再编译MXNet才能使用分布式训练机制。
通过调用`mxnet.kvstore.create`函数使用包含dist字符串的字符串参数来启用KVStore的分布式模式，
如：`kv = mxnet.kvstore.create('dist_sync')`

### Keys的分配
每个Server不一定要存储所有的keys和参数数组，参数可分布在不同的Server上，哪个Server存储特定的keys是随机决定的。KVStore会透明地处理不同Server上的keys分配。它会确保一个key被Pull时，该请求被发送到的Server有对应的Value。若某个keys的值非常大，它就会在不同Server上被分片。也就是说，不同Server会有不同的Value。分片阈值可以用环境变量`MXNET_KVSTORE_BIGARRAY_BOUND`来设置。

### 切分训练数据
此部分为MXNet的数据并行模式训练下使用。

单个Worker的数据并行训练，使用`mxnet.gluon.utils.split_and_load`来切分data iterator提供的样本，然后将它们加载到将要处理它们的设备上。

开始时我们需要将数据集分成n个部分，让每个worker获得不同的部分。随后，每个worker可以用`split_and_load`来将这部分的数据划分到单个机器的不同设备上。

>通常情况下，每个worker都是通过数据迭代器进行的数据拆分，通过传递切分的数量和切分部分的索引来迭代。 MXNet中支持此功能的一些迭代器是`mxnet.io.MNISTIterator`和`mxnet.io.ImageRecordIter`。如果你使用的是不同的迭代器，你可以看看上面的迭代器是如何实现此功能的。我们可以使用kvstore对象来获取当前worker的数量（kv.num_workers）和等级（kv.rank）。这些可以作为参数传递给迭代器。你可以看[example / gluon / image_classification.py](https://github.com/apache/incubator-mxnet/blob/master/example/gluon/image_classification.py)来查看一个示例用法。

### 更新权重
KVStore server提供两种模式：
1. 聚合梯度，并使用这些梯度来更新权重
2. 仅聚合梯度，不更新权重

后一种情况中，当一个Worker从KVStore中提取信息，它得到聚合后的梯度。随后Worker使用这些梯度并在本地应用权重。

当你使用Gluon时，你可以改变[Trainer](https://mxnet.cdn.apache.org/versions/1.7.0/api/python/docs/api/gluon/trainer.html)中的`update_on_kvstore`中的参数来切换这些模式：
```python
trainer = gluon.Trainer(net.collect_params(), optimizer='sgd',
						optimizer_params={'learning_rate':opt.lr,
										  'wd':opt.wd,
										  'momentum':opt.momentum,
										  'multi_precision':True},
						kvstore=kv,
						update_on_kvstore=True)
```
### 分布式训练的不同模式
需要在kvstore创建包含dist字段的字符串才会启用分布式。

以下是不同类型的kvstore：
1. `dist_sync`：同步分布式训练。所有worker在每批次计算时都使用同一组模型参数。

>每次批处理后，server在更新模型参数之前都会等待从每个worker上接收gradients。这种同步需要付出代价，因为worker必须等到server完成接收过程再开始拉取参数。在这种模式下，如果有worker崩溃，那么它会使所有worker的进度停止。
	
2.`dist_async`: 异步分布式训练。server从一个worker接受梯度后，立刻更新，以便用于未来的抓取。
>完成一批计算的worker可以从server中提取当前参数并开始下一批计算，即使其他worker尚未完成先前批的计算。这比`dist_sync`快，但可能需要更多的训练次数才能收敛。在异步模式下，需要传递优化器，因为在没有优化器的情况下，kvstore会用接收的权重替换存储的权重，这对于异步模式下的训练没有意义。权重的更新具有原子性，这意味着同一重量不会同时发生两次更新。但是，更新顺序无法保证。

3.`dist_sync_device`：与`dist_sync`相似，每个节点上使用多个GPU，在GPU上聚合梯度更新权重。此模式比`dist_sync`快但增加了GPU上的内存使用。

4.`dist_async_device`：与`dist_sync_device`相似，但处于异步。

### 梯度压缩
用来解决通信费用昂贵，计算时间与通信时间比例较低的问题。梯度压缩可以降低通信成本，从而加速训练。详情请参阅官方[梯度压缩文档](https://mxnet.cdn.apache.org/versions/1.7.0/api/faq/gradient_compression.html)。

PS：对于小型模型，由于通信和同步的开销，分布式训练有可能比单机训练速度慢。
 
## 分布式训练如何部署
### 配置集群
首先，配置服务器集群以便使用ssh启动分布式训练作业，若服务器需要使用密码进行身份验证，请为服务器建立信任关系，配置Linux服务器信任关系请参阅[此处](https://github.com/Jackxiini/Trust-relationship-configuration-between-Linux-servers/blob/main/%E6%93%8D%E4%BD%9C%E6%AD%A5%E9%AA%A4.md)查看详情。
### 环境变量及代码
我们可以使用launch.py脚本来进行分布式训练的部署，脚本可以从[此处](https://github.com/apache/incubator-mxnet/blob/master/tools/launch.py)获取。
我们仍然需要对代码进行修改以进行分布式训练。
#### 第一步：使用一个分布式KVStore
导入所需的MXNet包，按实际情况一般需要导入更多，此处只是为了方便后续例子的理解：
```python
import mxnet as mx
from mxnet import gluon
```
我们需要创建一个KVStore并让我们的`Trainer`使用它
```python
store = mx.kv.create('dist_async')
```
此处的`dist_async`可以用以上所述的其他字符串代替。

`trainer`的工作是获取在反向传递中计算出的梯度并更新模型的参数。 我们将告诉`trainer`在刚创建的KVStore中存储和更新参数。
```python
trainer = gluon.Trainer(net.collect_params(),
                        'adam', {'learning_rate': .001},
                        kvstore=store)
```

#### 第二步：分割训练集
在分布式训练中（使用数据并行），训练数据均分给所有worker，每个worker都使用其子集进行训练。 例如，如果我们有两台机器，每台机器都运行一个worker，每个worker管理多个GPU，我们将按如下所示拆分数据。 

注意: 我们不是根据GPU的数量来拆分数据，而是根据worker的数量来拆分数据。

![bj-d63ad1c12c44a3ac0c8f94600bd470ccccc6b733](https://user-images.githubusercontent.com/35672492/124058750-8f3e4000-da5c-11eb-9132-42347c539724.png)

每个worker可以知道集群中worker的总数，每个worker有一个rank号，取值范围是0到N-1，N是worker总数。

`store.num_workers`可以显示worker总数，`store.rank`可以显示worker的rank。

以下类可以加入代码以方便分割数据分配给worker使用：
```python
class SplitSampler(gluon.data.sampler.Sampler):
    """ 分割`num_parts`份的数据并给sample分配index `part_index`
    参数
    ----------
    length: int
      数据集的大小
    num_parts: int
      分割num_parts份的数据
    part_index: int
      需要读取的子集的index
    """
    def __init__(self, length, num_parts=1, part_index=0):
        # 计算子集的数据长度
        self.part_len = length // num_parts
        # 计算子集的起始index
        self.start = self.part_len * part_index
        # 计算子集的终止index
        self.end = self.start + self.part_len

    def __iter__(self):
        # 从`start`至`end`提取样本，打乱并返回它们
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len
```

然后我们可以使用`DataLoader`来使用`SplitSampler`，此处使用CIFAR10数据集作为案例:
```python
# 读取训练数据
train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True).transform(transform), 
								   batch_size, 
								   sampler=SplitSampler(50000, store.num_workers, store.rank))
```
#### 第三步：在多GPU上训练
之前提到，我们不会将数据集按照GPU数量分割，我们按照worker数来分割数据（通常等于机器数），worker将子集分割给其管理的各个GPU，再在各GPU间并行训练。
我们需要首先获取参与训练的GPU列表：
```python
ctx = [mx.gpu(i) for i in range(gpus_per_machine)]
```
然后批量训练，以下是一个例子：
```python
# 用多个GPU批量训练
def train_batch(batch, ctx, net, trainer):

    # 分割并读取数据进GPU
    data = batch[0]
    data = gluon.utils.split_and_load(data, ctx)

    # 分割并读取标签进GPU
    label = batch[1]
    label = gluon.utils.split_and_load(label, ctx)

    # 运行正向反向传递
    forward_backward(net, data, label)

    # 更新参数
    this_batch_size = batch[0].shape[0]
    trainer.step(this_batch_size)
```
此处是在多GPU上运行前向传递（计算loss）和反向传递（计算梯度）的代码：
```python
# 此处例子使用交叉熵损失函数，应按照实际情况修改
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# 在多GPU上运行一次前向和反向传递
def forward_backward(net, data, label):

    # 令autograd记忆正向传递
    with autograd.record():
        # 在所有GPU上计算loss
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]

    # 在所有GPU上做反向传递（计算梯度）
    for l in losses:
        l.backward()
```
使用我们定义的`train_batch`函数，就可以简单的训练一个epoch了：
```python
for batch in train_data:
    # 用多GPU训练批次数据
    train_batch(batch, ctx, net, trainer)
```
#### 最后一步：使用launch.py启动分布式训练
我们需要在多台服务器上启动多个进程。 每个主机上都需要启动一个worker和一个server，scheduler需要在其中一个主机上运行。

以下是在两台机器上运行分布式训练的命令例子：
```
python /home/xwj/test/launch.py -n 2 -s 2 -H hosts \
	--sync-dst-dir /home/xwj/dist --launcher ssh \
	"python /home/xwj/test/cifar10.py"
```
- `-n 2`指定有多少个worker被启动
- `-s 2`指定有多少个server被启动
- `-H hosts`指定用来进行分布式训练的集群中的主机列表
- `--sync-dst-dir`指定目标位置，在该位置将当前目录的内容进行同步
- `--launch ssh`让`launch.py`使用ssh在集群中的各个机器上登录并启动进程。选项还可有：
	- ssh 如果机器可以通过ssh进行通信而无需密码。 这是默认启动模式。
	- mpi 使用Open MPI时开启
	- sge 适用于Sun Grid引擎
	- yarn 适用于Apache yarn
	- local 用于在同一本地计算机上启动所有进程。 这可以用于调试。

- `"python /home/xwj/test/cifar10.py"`是将在每个启动的进程中执行的命令，此处内容是主程序的地址。

我们需要创建hosts文档来告诉launcher主机的IP地址：
```
~/dist$ cat hosts
```
在创建的hosts中写入各个主机的IP地址，例如：
```
10.157.6.182
10.157.6.183
```

最后运行命令，我们会看到以下的输出：
```
$ python /home/xwj/test/launch.py -n 2 -s 2 -H hosts --sync-dst-dir /home/xwj/dist --launcher ssh "python /home/xwj/test/cifar10.py"
2021-01-13 17:25:12,438 INFO rsync /home/xwj/test/ -> 10.157.6.183:/home/xwj/dist
2021-01-13 17:25:12,438 INFO rsync /home/xwj/test/ -> 10.157.6.182:/home/xwj/dist
[Epoch 0] Training: accuracy=0.125640
[Epoch 0] Validation: accuracy=0.102800
[Epoch 1] Training: accuracy=0.109440
[Epoch 1] Validation: accuracy=0.103200
[Epoch 2] Training: accuracy=0.108120
[Epoch 2] Validation: accuracy=0.113500
[Epoch 3] Training: accuracy=0.110960
[Epoch 3] Validation: accuracy=0.102800
[Epoch 4] Training: accuracy=0.106280
[Epoch 4] Validation: accuracy=0.102800
[Epoch 5] Training: accuracy=0.110440
[Epoch 5] Validation: accuracy=0.113500
```
