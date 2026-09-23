"""main：总指挥"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   #防止torch 与 numpy 的 OpenMP打架

from src.data import data_base
from src.predict import draw, predict, save_model
from src.train import train

#--------------------数据处理----------------------------
train_loader, test_loader, yte, pp, ncol = data_base()
#--------------------开始训练----------------------------
mt, losses, test_accs,train_accs = train(train_loader, test_loader, yte, in_dim=ncol)
#--------------------结果处理----------------------------
draw(losses, test_accs,train_accs)
#--------------------保存模型----------------------------
save_model(mt)
#--------------------进行预测----------------------------
m = input("是否开始预测（是，否）：")
if m == "是":
    predict(mt, pp)
    print("预测结束")
else:
    print("任务结束")
