#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trn_data=pd.read_csv('RegressoinTraining.csv')
x=np.array(trn_data['sales'])
y=np.array(trn_data['profit'])
trn_data.head()

test_data=pd.read_csv('RegressoinTesting.csv')
test_x=np.array(test_data['sales'])
test_y=np.array(test_data['profit'])
test_data.head()

plt.plot(x,y);
plt.plot(test_x,test_y);
plt.show()

#making an equation
m=np.size(x)
m_x=sum(x)
m_x2=sum(x*x)
m_y=sum(y)
m_yx=sum(y*x)

A= np.array([[m,m_x],[m_x,m_x2]])
B=np.array([[m_y],[m_yx]])

A_inv=np.linalg.inv(A)
theeta=A_inv.dot(B)

s=np.size(test_x)
new=np.zeros((s,1))
new[::]=1
test_x=test_x.reshape(s,1)
new=np.append(new,test_x,axis=1)


predict_y=new.dot(theeta)
predict_y=predict_y.reshape(32)


MSE=(1/(2*s))*sum((predict_y-test_y)**2)
print("MSE : ",MSE)


plt.plot(test_x,test_y);
plt.plot(test_x,predict_y);
plt.show()

######################
#quadratic
m=np.size(x)
m_x=sum(x)
m_x2=sum(pow(x,2))
m_x3=sum(pow(x,3))
m_x4=sum(pow(x,4))
m_y=sum(y)
m_yx=sum(y*x)
m_yx2=sum(y*pow(x,2))

A= np.array([[m,m_x,m_x2],[m_x,m_x2,m_x3],[m_x2,m_x3,m_x4]])
B=np.array([[m_y],[m_yx],[m_yx2]])


A_inv=np.linalg.inv(A)
theeta=A_inv.dot(B)

s=np.size(test_x)
new=np.zeros((s,1))
new[::]=1
test_x=test_x.reshape(s,1)
test_x2=np.power(test_x,2)
new=np.append(new,test_x,axis=1)
new=np.append(new,test_x2,axis=1)

predict_y=new.dot(theeta)
predict_y=predict_y.reshape(32)


MSE=(1/(2*s))*sum((predict_y-test_y)**2)
print("MSE : ",MSE)


plt.plot(test_x,test_y);
plt.plot(test_x,predict_y);
plt.show()

#################
##cubic
m=np.size(x)
m_x=sum(x)
m_x2=sum(pow(x,2))
m_x3=sum(pow(x,3))
m_x4=sum(pow(x,4))
m_x5=sum(pow(x,5))
m_x6=sum(pow(x,6))
m_y=sum(y)
m_yx=sum(y*x)
m_yx2=sum(y*pow(x,2))
m_yx3=sum(y*pow(x,3))


A= np.array([[m,m_x,m_x2,m_x3],[m_x,m_x2,m_x3,m_x4],[m_x2,m_x3,m_x4,m_x5],[m_x3,m_x4,m_x5,m_x6]])
B=np.array([[m_y],[m_yx],[m_yx2],[m_yx3]])


A_inv=np.linalg.inv(A)
theeta=A_inv.dot(B)

s=np.size(test_x)
new=np.zeros((s,1))
new[::]=1
test_x=test_x.reshape(s,1)
test_x2=np.power(test_x,2)
test_x3=np.power(test_x,3)
new=np.append(new,test_x,axis=1)
new=np.append(new,test_x2,axis=1)
new=np.append(new,test_x3,axis=1)

predict_y=new.dot(theeta)
predict_y=predict_y.reshape(32)


MSE=(1/(2*s))*sum((predict_y-test_y)**2)
print("MSE : ",MSE)


plt.plot(test_x,test_y);
plt.plot(test_x,predict_y);
plt.show()


# In[21]:





# In[25]:





# In[ ]:




