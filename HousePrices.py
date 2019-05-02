import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# creer un DataFrame à partir d'un csv 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


print ("Train shape:", train.shape)
print ("Test shape:", test.shape)
print (train.head())
print(train.SalePrice.describe())

train.Foundation.value_counts()
sns.countplot(train.Foundation)
plt.show()

plt.hist(train.SalePrice, color='blue')
plt.show()


plt.hist(np.log(train.SalePrice), bins = 25)
plt.show()


numeric_features = train.select_dtypes(include=[np.number])
correlation  = numeric_features.corr()

print (correlation ['SalePrice'].sort_values(ascending=False)[:5], '\n') #Afficher les 5 premieres features
print (correlation ['SalePrice'].sort_values(ascending=False)[-5:])

target = np.log(train.SalePrice)

plt.scatter(x=train['OverallQual'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('OverallQual')
plt.show()


plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('GrLivArea')
plt.show()


plt.scatter(x=train['GarageCars'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('GarageCars')
plt.show()
#il y a beaucoup de maison avec 0 GarageCars ==> Donnée aberrante
#supp GarageCars
train = train[train['GarageArea'] < 1200]

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # onforce la même échelle qu'avant
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


#on examine les valeurs nulles (NaN)
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)



#non-numeriques features

categoricals = train.select_dtypes(exclude=[np.number])
print(categoricals.describe())


print ("Original: \n") 
print (train.Street.value_counts(), "\n")

#transformer les valeurs de enc_street (il y en a 5) en Boolean en utilisant pd.get_dummies() 
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

print ('Encode: \n') 
print (train.enc_street.value_counts())


condition_pivot = train.pivot_table(index='SaleCondition',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


#on explore avec plot.

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# on rempli les valeurs manquantes avec une valeur moyenne
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
#on vérifie
print(sum(data.isnull().sum() != 0))



y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha aide à montrer les données qui se chevauchent
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


