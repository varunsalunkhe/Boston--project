import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


from sklearn.datasets import load_boston
da =load_boston()

da.keys()

da.feature_names

dataset =pd.DataFrame(data = da.data, columns = da.feature_names)
dataset['prize']=da.target
dataset.head()

dataset.isnull().mean()

dataset.corr()

plt.subplots(figsize=(10, 7))
sns.heatmap(dataset.corr(), annot=True, cmap='bwr')

import warnings 
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.distplot(dataset.TAX);

x=dataset.loc[: , dataset.columns!='prize']
y=dataset.prize

#standarization of data
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x= pd.DataFrame(data = scale.fit_transform(x), columns = da.feature_names)

x.head()

# spliting train and test data
from sklearn.model_selection import train_test_split
train_x, test_x, train_y , test_y = train_test_split(x, y, test_size=0.3)
train_x.shape, test_x.shape, train_y.shape , test_y.shape

# training model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)

pred=model.predict(test_x)
print(model.score(test_x, pred))

from sklearn.metrics import mean_squared_error
mean_squared_error(pred, test_y)

plt.scatter(pred, test_y)

model.predict(scale.transform(da.data[0].reshape(1,-1)))

#saving model
import joblib
joblib.dump(model,"model.pkl")

joblib.dump(scale,'scale.pkl')

