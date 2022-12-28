import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Data Mining/Homework/Bank_Customer.csv', sep=',', header=0)

print(data.describe(include='all'))

sns.set_theme()

count=[data.churn[data['churn']==1].count(),data.churn[data['churn']==0].count()]
fig, ax1 = plt.subplots(figsize=(5, 5))
ax1.pie(count, labels=[1,0], autopct='%.1f%%')
plt.title('churn')
plt.savefig('Data Mining/Homework/Pic/over_all.png')
plt.close()

fig,axar=plt.subplots(3,2,figsize=(15,12))
sns.boxplot(x='churn',y='credit_score',hue='churn',data=data,ax=axar[0][0])
sns.boxplot(x='churn',y='age',hue='churn',data=data,ax=axar[0][1])
sns.boxplot(x='churn',y='tenure',hue='churn',data=data,ax=axar[1][0])
sns.boxplot(x='churn',y='balance',hue='churn',data=data,ax=axar[1][1])
sns.boxplot(x='churn',y='products_number',hue='churn',data=data,ax=axar[2][0])
sns.boxplot(x='churn',y='estimated_salary',hue='churn',data=data,ax=axar[2][1])
plt.savefig('Data Mining/Homework/Pic/nominal.png')
plt.close()

fig,axar = plt.subplots(2,2,figsize=(10,7))
sns.countplot(x='country',hue='churn',data=data,ax=axar[0][0])
sns.countplot(x='gender',hue='churn',data=data,ax=axar[0][1])
sns.countplot(x='credit_card',hue='churn',data=data,ax=axar[1][0])
sns.countplot(x='active_member',hue='churn',data=data,ax=axar[1][1])
plt.savefig('Data Mining/Homework/Pic/not_nominal.png')
plt.close()