import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# %%
print(train_data.columns)
# %%
# Cabin 船艙號碼
# Parch 在船上同為家族的父母及小孩的數目
# SibSp 在船上同為兄弟姐妹或配偶的數目
# %%
print(train_data.info())
# %%
print(train_data)
# %%
# 資料清洗
train_data_drop = train_data.copy()
train_data_drop.dropna(subset=['Age'], inplace=True)
print(train_data_drop)
# %%
features = ['PassengerId', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Fare']
X = train_data_drop[features]
y = train_data_drop['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train_std = st.fit_transform(X_train)
X_test_std = st.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train_std, y_train)
y_hat = model.predict(X_test_std)
print('直接刪掉空值 Accuarcy: ', accuracy_score(y_hat, y_test))
# %%
train_data_median = train_data.copy()
train_data_median['Age'] = train_data_median['Age'].fillna(train_data['Age'].median())
print(train_data_median)
# %%
features = ['PassengerId', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Fare']
X = train_data_median[features]
y = train_data_median['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train_std = st.fit_transform(X_train)
X_test_std = st.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train_std, y_train)
y_hat = model.predict(X_test_std)
print('用中位數填空值 Accuarcy: ', accuracy_score(y_hat, y_test))
# %%
# 選定用中位數處理Age缺失值部分
train_data = train_data_median
print(train_data)
# %%
# 官方解釋1是頭等艙，2是二等艙，3是三等艙
pclass_num_of_Survivors = train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(pclass_num_of_Survivors)
# %%
import matplotlib.pyplot as plt

plt.bar(pclass_num_of_Survivors['Pclass'], pclass_num_of_Survivors['Survived'])
plt.xlabel('Pclass')
plt.ylabel('Survived')
#限制x軸的範圍
plt.xticks(pclass_num_of_Survivors['Pclass'])
# 船艙等級越高，存活率越高
# %%
sex_num_of_Survivors = train_data[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)
sex_num_of_Survivors
plt.bar(sex_num_of_Survivors['Sex'], sex_num_of_Survivors['Survived'])
plt.show()
# %%
import seaborn as sns
sns.heatmap(train_data.corr())
# %%
train_data['Sex'].replace({'male':1, 'female':0}, inplace=True)
#1代表男性 0代表女性
print(train_data)
# %%
features = ['PassengerId', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Fare']
X = train_data[features]
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# %%
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print('KNeighbors Accuarcy:', accuracy_score(y_hat, y_test))

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print('DecisionTreeClassifier Accuarcy:', accuracy_score(y_hat, y_test))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print('RandomForestClassifier Accuarcy:', accuracy_score(y_hat, y_test))

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(random_state=0)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print('MLPClassifier Accuarcy:', accuracy_score(y_hat, y_test))

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=0)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
print('AdaBoostClassifier Accuarcy:', accuracy_score(y_hat, y_test))
# %%
# 針對決策叢林做優化
for n_estimator in range(10, 100, 10):
    model = RandomForestClassifier(n_estimators=n_estimator, random_state=0)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    print('n_estimator:', n_estimator, 'Accuarcy:', accuracy_score(y_hat, y_test))
# %%
from sklearn.metrics import roc_curve, auc
y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
print('AUC: ', auc(fpr, tpr))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()