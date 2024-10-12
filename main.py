import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

df=pd.read_csv("C:\\Users\\Admin\\Downloads\\SolarPrediction.csv")
df

df.describe(include='all')

df.info()

df.shape

df.isnull().sum()

df.drop(columns=['Data','Time'],inplace=True) #droping the columns which are not required

df

#changing the data in datetime format
df['TimeSunRise'] = pd.to_datetime(df['TimeSunRise'], format='%H:%M:%S') 
df['TimeSunSet'] = pd.to_datetime(df['TimeSunSet'], format='%H:%M:%S')

df['TSRhour'] = df['TimeSunRise'].dt.hour.astype(int)
df['TSRmin'] = df['TimeSunRise'].dt.minute.astype(int)
df['TSShour'] = df['TimeSunSet'].dt.hour.astype(int)
df['TSSmin'] = df['TimeSunSet'].dt.minute.astype(int)
df.drop(columns=['TimeSunRise','TimeSunSet'],inplace=True)
df

df.info()

Y = df[['Radiation']]
X = df.drop(columns=['Radiation'])

X

Y

plt.plot(Y)
plt.show

X.info()
Y.info()

## AllFeatures

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42,shuffle=True)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=25, random_state=3)
regr.fit(x_train, y_train)

Allfeatures=regr.score(x_train, y_train)
Allfeatures

regr.score(x_test, y_test)

## VarianceThreshold

x_train_1,x_test_1,y_train_1,y_test_1 = x_train.copy(),x_test.copy(),y_train.copy(),y_test.copy() 

x_train_1.var(axis=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_x_train_1= scaler.fit_transform(x_train_1)

fig,ax=plt.subplots()

x=X.columns
y=scaled_x_train_1.var(axis=0)

ax.bar(x,y,width=0.8)
ax.set_xlabel('Features')
ax.set_ylabel('Variance')
ax.set_ylim(0,0.1)

for index, value in enumerate(y):
    plt.text(x=index,y=value+0.001,s=str(round(value, 3)),ha='center')

fig.autofmt_xdate()
plt.tight_layout()

from sklearn.metrics import f1_score

sel_x_train_1=x_train_1.drop(['Speed','TSRhour','Pressure'],axis=1)
sel_x_test_1=x_test_1.drop(['Speed','TSRhour','Pressure'],axis=1)
sel_y_train_1=x_train_1.drop(['Speed','TSRhour','Pressure'],axis=1)
sel_y_test_1=x_test_1.drop(['Speed','TSRhour','Pressure'],axis=1)

regr.fit(sel_x_train_1,sel_y_train_1)

varianceScore=regr.score(sel_x_train_1,sel_y_train_1)
varianceScore

regr.score(sel_x_test_1,sel_y_test_1)

fig,ax=plt.subplots()

x=['All features','Variance']
y=[Allfeatures,varianceScore]

ax.bar(x,y,width=0.6)
ax.set_xlabel('Feature selection methods')
ax.set_ylabel('Score')
ax.set_ylim(0,1.1)

for index, value in enumerate(y):
    plt.text(x=index,y=value+0.001,s=str(round(value, 3)),ha='center')

fig.autofmt_xdate()
plt.tight_layout()