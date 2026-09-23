"""train：训练模型，返回训练好的模型和两条曲线的数字数据"""

'''命名解释：
mt:model train简称，用来导入模型
loss_fn:算交叉熵损失的函数，输入真实值和预测值
opt:Adam优化器函数
losses：Loss的数据
test_accs、acc：测试集的精确率
predi：预测结果'''

import torch
import torch.nn as nn
from src.model import Net #各个文件之间可以通过import来交接类或函数

def train(train_loader, test_loader, yte, in_dim=13, epochs=200, lr=0.01):

    torch.manual_seed(42)#固定种子
    #如果使用GPU要写  torch.cuda.manual_seed_all(42)

    mt = Net(in_dim)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mt.parameters(), lr=lr)
    losses, test_accs,train_accs = [], [],[]

    for epoch in range(epochs):
        mt.train()#切换训练模式，主要是开启dropout
        total_loss, total_n = 0.0, 0
        for xb, yb in train_loader:
            out = mt(xb)#算输出
            loss = loss_fn(out, yb)

            opt.zero_grad()#清除梯度
            loss.backward()#反向传播算梯度
            opt.step()#正式工作

            total_loss += loss.item() * len(yb)
            total_n += len(yb)

        losses.append(total_loss / total_n)#.item（）用来把张量转成数字(画图会用)

        mt.eval()#关闭训练模式，主要是关闭dropout（不然测试集结果不准）
        with torch.no_grad():#with自动资源释放,不用在意开关
            correct, total = 0, 0
            for xb, yb in test_loader:
                predi = mt(xb).argmax(1)#挑出每个乘客分数更高的那一类，得到 0 或 1（分数是模型的输出，不是概率，没有和为一，要知道概率，用softmax）
                correct += (predi == yb).sum().item()
                total += len(yb)
            acc = correct / total
            #这里的mean()刚好可以算出准确率
            test_accs.append(acc)

        with torch.no_grad():#with自动资源释放,不用在意开关
            correct, total = 0, 0
            for xb, yb in train_loader:
                predi = mt(xb).argmax(1)#挑出每个乘客分数更高的那一类，得到 0 或 1（分数是模型的输出，不是概率，没有和为一，要知道概率，用softmax）
                correct += (predi == yb).sum().item()
                total += len(yb)
            acc = correct / total
            #这里的mean()刚好可以算出准确率
            train_accs.append(acc)

        if (epoch + 1) % 20 == 0 or epoch + 1 ==1:
            print('第', epoch + 1, '轮 | 损失', round(losses[-1], 4),
                  '| 测试准确率', round(acc, 4))#保留4位小数

    print('最终测试准确率:', round(test_accs[-1], 4))#【-1】是最后一次
    print('盲猜遇难的概率:', round((yte == 0).mean(), 4))

    return mt, losses, test_accs,train_accs
