mport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import seaborn as sns
#dataset in dataframe
col=["flenght","fwidth","fsize","fconc","fconc1","fasym","fm3long","fm3trans","falpha","fdist","class1"]
df=pd.read_csv("magic04.data",names=col)
df.head()
df["class1"]=(df["class1"]=="g").astype(int)
# Spliting the DataFrame based on the 'class1' column
class1_df = df[df['class1'] == 1]
class0_df = df[df['class1'] == 0]
# Calculate the mean and median for each column based on 'class1' value
mean_class1 = np.mean(class1_df)
median_class1 = class1_df.median()
mean_class0 = np.mean(class0_df)
median_class0 = class0_df.median()
# Display the results
print("Mean and Median for class1 (class1=1):")
print(mean_class1)
print(median_class1)
print("\nMean and Median for class0 (class1=0):")
print(mean_class0)
print(median_class0)
# Calculate the variance and standard devietion for each column based on 'class1' value
var_class1 = np.var(class1_df)
std_class1 = np.std(class1_df)
var_class0 = np.var(class0_df)
std_class0 = np.std(class0_df)
# Display the results
print("variance and standard devietion for class1 (class1=1):")
print(var_class1)
print(std_class1)
print("\nvariance and standard devietion for class0 (class1=0):")
print(var_class0)
print(std_class0)
for label in col[:-1]:
  plt.hist(df[df["class1"]==1][label],color='blue',label="gamma",alpha=0.5,density=True)
  plt.hist(df[df["class1"]==0][label],color='green',label="hardron",alpha=0.5,density=True)
  plt.title(label)
  plt.ylabel("probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()
df.info()
df.describe()
df.isnull().sum()
sns.pairplot(df)
corr = df.corr()
plt.figure(figsize=(10, 8))

ax = sns.heatmap(corr, vmin = -1, vmax = 1, annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
corr
train,valid,test=np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])
def scale_dataset(dataframe,oversample=False):
  x=dataframe[dataframe.columns[:-1]].values
  y=dataframe[dataframe.columns[-1]].values
  scaler=StandardScaler()
  x=scaler.fit_transform(x)
  if oversample:
    ros=RandomOverSampler()
    x,y=ros.fit_resample(x,y)
  data=np.hstack((x,np.reshape(y,(-1,1))))
  return data,x,y
train,x_train,y_train =scale_dataset(train,oversample=True)
vaild,x_vaild,y_vaild=scale_dataset(valid,oversample=False)
test,x_test,y_test=scale_dataset(test,oversample=False)
lb_model=LogisticRegression()
lb_model=lb_model.fit(x_train,y_train)
y_pred=lb_model.predict(x_test)
print(classification_report(y_test,y_pred))
knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train,y_train)
y_pred=knn_model.predict(x_test)
print(classification_report(y_test,y_pred))
nb_model=GaussianNB()
nb_model=nb_model.fit(x_train,y_train)
y_pred=nb_model.predict(x_test)
print(classification_report(y_test,y_pred))
b_model=RandomForestClassifier()
rb_model=rb_model.fit(x_train,y_train)
y_pred=rb_model.predict(x_test)
print(classification_report(y_test,y_pred))
ab_model=AdaBoostClassifier()
ab_model=ab_model.fit(x_train,y_train)
y_pred=ab_model.predict(x_test)
print(classification_report(y_test,y_pred))
dt_model=DecisionTreeClassifier()
dt_model=dt_model.fit(x_train,y_train)
y_pred=dt_model.predict(x_test)
print(classification_report(y_test,y_pred))
svm_model=SVC()
svm_model=svm_model.fit(x_train,y_train)
y_pred=svm_model.predict(x_test)
print(classification_report(y_test,y_pred))
