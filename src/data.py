"""data：读数据 + 预处理 + 分层划分 + 转张量，一次性全部 return 出去"""

'''
部分命名说明：
Xtr：除survived外特征的训练集
ytr：survived特征的训练集
Xte：除survived外特征的测试集
yte：survived特征的测试集


'''
import os
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import TitanicDataset
from src.preprocess import PreProcess, ORDER

def data_base():
    TEST_RATE = 0.2
    PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#对绝对路径（abspath）删最近的地址（dirname）（得到总文件所在地址）
    DATA = os.path.join(PATH, "data", 'titanic.csv')
    df = pd.read_csv(DATA)

    a = PreProcess()

    df2 = df.copy()  #不动原数据，复制一个
    df2 = a.newf(df2)
    df2 = a.d(df2)

    te, tr = a.split(df, TEST_RATE, 42)

    a.fit(df2.iloc[tr])
    df2 = a.fill(df2)
    df2 = a.recode(df2)
    a.fit_scale(df2.iloc[tr])
    df2 = a.one(df2)

    b = df2.isnull().sum().astype(bool)#isnull()：找缺失（False = 没有缺失）
    print('-----------------------------------')
    print('检查缺失值（False = 没有缺失）：')
    print(b.to_string())  #.to_string()：别给我省略号，整张表全打出来"
    print('检查类型（应全是数字）：')
    print(df2.dtypes.to_string())
    print('-----------------------------------')

    X = df2[ORDER].values.astype('float32')
    y = df['Survived'].values.astype('int64')

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    print('训练集', len(Xtr), '条, 测试集', len(Xte), '条')
    print('训练集幸存率', round(ytr.mean(), 3), ', 测试集幸存率', round(yte.mean(), 3))

    train_ds = TitanicDataset(Xtr, ytr)
    test_ds = TitanicDataset(Xte, yte)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))

    return train_loader, test_loader, yte, a, len(ORDER)
