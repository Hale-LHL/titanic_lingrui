"""PreProcess：所有的数据清洗、编码、缩放都在这里"""

'''部分命名说明：
d:drop的简称，删特征
ONE：要标准化的特征
ORDER：顺序特征
newf：new feature,造新特征
one:标准化'''

import numpy as np

ONE = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']         #要标准化的特征
ORDER = [
    'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone',
    'Sex_male',
    'Pclass_1', 'Pclass_2', 'Pclass_3',
    'Embarked_C', 'Embarked_Q', 'Embarked_S']              #顺序特征


class PreProcess():        #各个def括号里的和return基本相同
    def fit(self, df):                 #算预参数（由数据决定，与参数不同，和训练无关）
        self.age_median = df["Age"].median()
        self.fare_median = df["Fare"].median()
        self.embarked_mode = df["Embarked"].mode()[0]      #如果有多个众数，选第一个
        return df

    def fit_scale(self, df):                    #标准化
        self.mean = df[ONE].mean()
        self.std = df[ONE].std().replace(0, 1.0)
        return df

    def one(self, df):
        df[ONE] = (df[ONE] - self.mean) / self.std
        return df

    def d(self, df):
        df = df.drop(columns=["PassengerId", "Name", "Survived", "Ticket", "Cabin"])
        return df

    def fill(self, df):
        df["Age"] = df["Age"].fillna(self.age_median)
        df["Fare"] = df["Fare"].fillna(self.fare_median)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)
        return df

    def recode(self, df):             #独热编码
        df["Embarked_C"] = (df["Embarked"] == "C").astype(int)
        df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
        df["Embarked_S"] = (df["Embarked"] == "S").astype(int)
        df["Pclass_1"] = (df["Pclass"] == 1).astype(int)
        df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
        df["Pclass_3"] = (df["Pclass"] == 3).astype(int)
        df["Sex_male"] = (df["Sex"] == "male").astype(int)
        df = df.drop(columns=["Embarked", "Pclass", "Sex"])
        return df

    def newf(self, df):
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        return df

    def split(self, df, test_size=0.2, seed=42):
        np.random.seed(seed)
        dfs = df['Survived'].values.astype('int64')
        surv = np.where(dfs == 1)[0]#[0]才能得到元组中的下标
        dead = np.where(dfs == 0)[0]
        np.random.shuffle(surv)
        np.random.shuffle(dead)
        s_te_idx = int(len(surv) * test_size)
        d_te_idx = int(len(dead) * test_size)#注意转成整型
        te = np.concatenate([surv[:s_te_idx], dead[:d_te_idx]])
        tr = np.concatenate([surv[s_te_idx:], dead[d_te_idx:]])#外层也有个【】
        np.random.shuffle(te)
        np.random.shuffle(tr)
        return te, tr
