#!/usr/bin/env python
# coding: utf-8

# # 1-Importation et explorationdes données

# Dans cette Partie on va explorer les données,on cherche est ce qu'il y a des données manquantes,c'est quoi la tendance generale et la relation entre les variables.

# In[56]:


import matplotlib.pyplot as plt


# In[57]:


import pandas as pd
data = pd.read_csv('carData.csv')


# In[58]:


data


# In[189]:


data['Transmission']


# In[59]:


data.shape
data.columns


# In[60]:


data['Seller_Type']


# In[61]:


data.describe


# In[62]:


data.describe(include = "all")


# In[63]:


data.info()


# In[64]:


data['Car_Name'].value_counts()


# In[65]:


missing_data = data.isnull()
missing_data.head(5)


# In[66]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# 

# In[67]:


data['Selling_Price'].describe


# In[68]:


data['Present_Price'].astype(int)


# In[69]:


data['Present_Price'].describe()


# In[70]:


data['Present_Price'].astype(int)
data['Selling_Price'].astype(int)
data['Kms_Driven'].astype(int)
data['Year'].astype(int)
data['Owner'].astype(int)


# On se rends compte qu'il n' y a pas des données manquantes,on a convertis les variables numeriques étaient considerées comme objet à des variables réelles

# In[71]:


data.describe()


# In[72]:


print(data.dtypes.value_counts())


# In[73]:


data.hist(column='Present_Price')


# On constate que plus que la voiture est chere plus qu'il y a moins de voiture,ce qui est une tendance normale,d'ou l'échantillon est representative.
# 

# In[74]:


data.hist(column='Selling_Price')


# In[75]:


var = data.groupby('Fuel_Type').Present_Price.mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Fuel_Type')
ax1.set_ylabel('Prix moyen')
ax1.set_title("Fuel_Type Vs Price")
var.plot(kind='bar')


# On constate qu'il y a plus de voitures dans notre échantillon qui fonctione en Diesel,ce qui est compatible avec la realité.

# In[192]:


var = data.groupby('Year').Selling_Price.mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Year')
ax1.set_ylabel('Prix moyen')
ax1.set_title("Year Vs Price")
var.plot(kind='bar')


# On constate que le prix moyen des voitures changent avec l'ancienneté,ce qui est tout à fait normale

# In[193]:


var = data.groupby('Transmission').Selling_Price.mean()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Seller_Type')
ax1.set_ylabel('Prix moyen')
ax1.set_title("Seller_Type Vs Price")
var.plot(kind='bar')


# Les voitures automatiques sont en moyenne plus cheres que manuelle ce qui est logique,parceque les voitures automatiques sont plus neuves.

# In[79]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[81]:


sns.catplot(x="Fuel_Type", y="Present_Price", jitter=False, data=data)


# In[ ]:


sns.catplot(x="Car_Name", y="Present_Price", jitter=False, data=data)


# In[83]:


sns.catplot(x="Car_Name", y="Present_Price", jitter=False, data=data)


# In[84]:


data.corr()


# In[86]:


corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# A l'exception des variables Present_Price et Selling_Price qui ont fortement correlé,il y a une faible correlation entre les autres variables.

# In[194]:


data.insert(2, "Age", 2020-data['Year'], True)


# In[88]:


data


# In[89]:


data.corr()


# In[90]:


corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[157]:


X1 = data['Age'].to_numpy()
Y = data['Present_Price'].to_numpy()


# In[160]:


X2 = data[['Age','Kms_Driven']].to_numpy()


# In[94]:


from pylab import *
m,b = polyfit(X1, Y, 1) 

plot(X1, Y, 'yo', X1, m*X1+b, '--k') 
show()


# In[96]:


m,b


# La fonction Polyfit sur numpy permet d'effectuer la regression lineaire,en fait c'est une fonction polynomial et le polynome d'ordre 1 correspend à la regression lineaire simple.

# In[100]:


import scipy
lr = scipy.stats.linregress(X1,Y)


# In[101]:


print(lr)


# In[137]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


    


# In[139]:


X1


# In[140]:


Y


# In[196]:


reg=regressor.fit(X1.reshape(-1, 1),Y)


# In[146]:


print(reg.intercept_, reg.coef_)


# In[197]:


y_pred = reg.predict(X1.reshape(-1, 1)) 
plt.scatter(X1.reshape(-1, 1), Y, color ='b') 
plt.plot(X1.reshape(-1, 1), y_pred, color ='k') 
  
plt.show()


# In[162]:


reg1=regressor.fit(X2.reshape(-1, 2),Y)


# In[163]:


print(reg1.intercept_, reg1.coef_)


# In[167]:


X3=X2.reshape(-1, 2)[:, 0]


# In[170]:


X3.shape


# In[171]:


Y.shape


# In[147]:


def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 


# In[148]:


def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 


# C'est dessus une fonction qui estime les parametres de la regression linéaire simples.

# In[149]:


estimate_coef(X1, Y)


# In[152]:


def main(): 
    # observations 
    x = X1
    y = Y
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb_0 = {}  \ \nb_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main() 


# In[150]:


plot_regression_line(X1, Y,estimate_coef(X1, Y))


# In[180]:


from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X1.reshape(-1, 1),Y)


# In[181]:


y_pred1 = regressor.predict(X1.reshape(-1, 1)) 
plt.scatter(X1.reshape(-1, 1), Y, color ='b') 
plt.plot(X1.reshape(-1, 1), y_pred1, color ='k') 
  
plt.show()


# In[183]:


regressor.get_params()


# In[185]:


regressor.score(X1.reshape(-1, 1),Y)

