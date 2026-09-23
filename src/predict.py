"""predict：画图、存模型、预测新乘客"""

import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from src.preprocess import ORDER   #函数外的值直接导import，函数里的用return


def draw(losses, test_accs,train_accs):
    plt.plot(losses)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('Training_loss_curve.png', dpi=120)
    plt.close()  #要关掉，不然要覆盖

    plt.plot(test_accs,label='test',marker='o')
    plt.plot(train_accs,label='train',marker='s')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.savefig('Test_accuracy_curve.png', dpi=120)
    plt.close()
    print('曲线已保存')


def save_model(mt):
    os.makedirs('model', exist_ok=True)
    torch.save(mt.state_dict(), 'model/titanic.pth')
    print('模型已保存到 model/titanic.pth')


def predict(mt, pp):#mt,pp两个对象不用引入到这个文件，再main中给他就是了
    print('开始预测')

    m = input('是否使用默认数据（一等舱25岁女性）？填 是 或 否：')

    if m == '是':
        PassengerId = 999
        Pclass = 1
        Name = 'Demo, Miss. Lily'
        Sex = 'female'
        Age = 25.0
        SibSp = 0
        Parch = 0
        Ticket = 'UNKNOWN'
        Fare = 100.0
        Cabin = None
        Embarked = 'S'
        print('已使用默认乘客：Demo, Miss. Lily，一等舱25岁')
    else:
        PassengerId = int(input('PassengerId 乘客编号（随便填，不影响结果）：'))
        Pclass = int(input('Pclass 舱位（填 1 或 2 或 3）：'))
        Name = input('Name 姓名（随便填，不影响结果）：')
        Sex = input('Sex 性别（填 male 或 female）：')
        Age = float(input('Age 年龄：'))
        SibSp = int(input('SibSp 同行的兄弟姐妹配偶数：'))
        Parch = int(input('Parch 同行的父母子女数：'))
        Ticket = input('Ticket 票号（随便填，不影响结果）：')
        Fare = float(input('Fare 票价：'))
        Cabin = input('Cabin 舱位号（随便填，不影响结果）：')
        Embarked = input('Embarked 登船港（填 C 或 Q 或 S）：')

    raw = {'PassengerId': PassengerId, 'Survived': None, 'Pclass': Pclass,
           'Name': Name, 'Sex': Sex, 'Age': Age, 'SibSp': SibSp,
           'Parch': Parch, 'Ticket': Ticket, 'Fare': Fare, 'Cabin': Cabin,
           'Embarked': Embarked}

    new_df = pd.DataFrame([raw])
    new_df = pp.newf(new_df)
    new_df = pp.d(new_df)
    new_df = pp.fill(new_df)
    new_df = pp.recode(new_df)
    new_df = pp.one(new_df)
    new_one = new_df[ORDER].values.astype('float32')

    mt.eval()
    with torch.no_grad():
        out = mt(torch.tensor(new_one))
        probi = torch.softmax(out, dim=1)[0, 1].item()

    if probi > 0.5:
        result = '幸存'
    else:
        result = '遇难'

    if Sex == 'male':
        sex = '男'
    else:
        sex = '女'
    print('%s等舱%s岁%s性预测%s，概率为 %.4f' % (Pclass, Age, sex, result, probi))
    return probi
