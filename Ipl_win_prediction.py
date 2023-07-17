#!/usr/bin/env python
# coding: utf-8

# In[16]:


import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[17]:


df=pd.read_csv("C:\computer science engineering\ML\IPL Score Prediction\ipl.csv")


# In[18]:


df


# In[30]:


# Check if columns exist in DataFrame
columns_to_drop = ['mid', 'batsman', 'bowler', 'striker', 'non-striker']
existing_columns = set(df.columns)
columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

# Drop columns if they exist
if columns_to_drop:
    df.drop(columns_to_drop, axis=1, inplace=True)
    print(f"Dropped columns: {columns_to_drop}")
else:
    print("No columns found in the DataFrame.")

df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))


# In[31]:


df


# In[33]:


df=df[df['overs']>=5.0]


# In[34]:


df


# In[35]:


def f(x):
    if x=='M Chinnaswamy Stadium':
        return 'M Chinnaswamy Stadium, Bangalore'
    elif x=='Feroz Shah Kotla':
        return 'Feroz Shah Kotla, Delhi'
    elif x=='Wankhede Stadium':
        return 'Wankhede Stadium, Mumbai'
    elif x=='Sawai Mansingh Stadium':
        return 'Sawai Mansingh Stadium, Jaipur'
    elif x=='Eden Gardens':
        return 'Eden Gardens, Kolkata'
    elif x=='Dr DY Patil Sports Academy':
        return 'Dr DY Patil Sports Academy, Mumbai'
    elif x=='Himachal Pradesh Cricket Association Stadium':
        return 'Himachal Pradesh Cricket Association Stadium, Dharamshala'
    elif x=='Subrata Roy Sahara Stadium':
        return 'Maharashtra Cricket Association Stadium, Pune'
    elif x=='Shaheed Veer Narayan Singh International Stadium':
        return 'Raipur International Cricket Stadium, Raipur'
    elif x=='JSCA International Stadium Complex':
        return 'JSCA International Stadium Complex, Ranchi'
    elif x=='Maharashtra Cricket Association Stadium':
        return 'Maharashtra Cricket Association Stadium, Pune'
    elif x=='Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':
        return 'ACA-VDCA Stadium, Visakhapatnam'
    elif x=='Punjab Cricket Association IS Bindra Stadium, Mohali':
        return 'Punjab Cricket Association Stadium, Mohali'
    elif x=='Holkar Cricket Stadium':
        return 'Holkar Cricket Stadium, Indore'
    elif x=='Sheikh Zayed Stadium':
        return 'Sheikh Zayed Stadium, Abu-Dhabi'
    elif x=='Sharjah Cricket Stadium':
        return 'Sharjah Cricket Stadium, Sharjah'
    elif x=='Dubai International Cricket Stadium':
        return 'Dubai International Cricket Stadium, Dubai'
    elif x=='Barabati Stadium':
        return 'Barabati Stadium, Cuttack'
    else:
        return x


# In[36]:


neglact_ground = ['Newlands', "St George's Park",
                    'Kingsmead', 'SuperSport Park', 'Buffalo Park',
                    'New Wanderers Stadium', 'De Beers Diamond Oval',
                    'OUTsurance Oval', 'Brabourne Stadium']


# In[38]:


df=df[True^(df['venue'].isin(neglact_ground))]
df['venue']=df['venue'].apply(f)


# In[39]:


df_new=pd.get_dummies(data=df,columns=['venue','bat_team','bowl_team'])


# In[40]:


df_new = df_new[['date','venue_ACA-VDCA Stadium, Visakhapatnam',
       'venue_Barabati Stadium, Cuttack', 'venue_Dr DY Patil Sports Academy, Mumbai',
       'venue_Dubai International Cricket Stadium, Dubai',
       'venue_Eden Gardens, Kolkata', 'venue_Feroz Shah Kotla, Delhi',
       'venue_Himachal Pradesh Cricket Association Stadium, Dharamshala',
       'venue_Holkar Cricket Stadium, Indore',
       'venue_JSCA International Stadium Complex, Ranchi',
       'venue_M Chinnaswamy Stadium, Bangalore',
       'venue_MA Chidambaram Stadium, Chepauk',
       'venue_Maharashtra Cricket Association Stadium, Pune',
       'venue_Punjab Cricket Association Stadium, Mohali',
       'venue_Raipur International Cricket Stadium, Raipur',
       'venue_Rajiv Gandhi International Stadium, Uppal',
       'venue_Sardar Patel Stadium, Motera',
       'venue_Sawai Mansingh Stadium, Jaipur',
       'venue_Sharjah Cricket Stadium, Sharjah',
       'venue_Sheikh Zayed Stadium, Abu-Dhabi',
       'venue_Wankhede Stadium, Mumbai','bat_team_Chennai Super Kings',
       'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
       'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians',
       'bat_team_Rajasthan Royals', 'bat_team_Royal Challengers Bangalore',
       'bat_team_Sunrisers Hyderabad','bowl_team_Chennai Super Kings',
       'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
       'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians',
       'bowl_team_Rajasthan Royals', 'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'total']]


# In[41]:


df_new.reset_index(inplace=True)
df_new.drop('index',inplace=True,axis=1)

df_new.head(2)


# In[42]:


df_new.shape


# In[45]:


scaler=StandardScaler()
scaled_columns=scaler.fit_transform(df_new[['runs','wickets','overs','runs_last_5','wickets_last_5']])

pickle.dump(scaler,open('scaler.pkl','wb'))


# In[52]:


scaled_columns=pd.DataFrame(scaled_columns,columns=['runs','wickets','overs','runs_last_5','wickets_last_5'])
df_new.drop(['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5'],axis=1,inplace=True)
df_new = pd.concat([df_new,scaled_columns],axis=1)


X_train = df_new.drop('total',axis=1)[df_new['date'].dt.year<=2016]
X_test = df_new.drop('total',axis=1)[df_new['date'].dt.year>=2017]

X_train.drop('date',inplace=True,axis=1)
X_test.drop('date',inplace=True,axis=1)


y_train = df_new[df_new['date'].dt.year<=2016]['total'].values
y_test = df_new[df_new['date'].dt.year>=2017]['total'].values


# In[53]:


df_new


# In[54]:


df_new.shape


# In[56]:


##Ridge Regressor

ridge=Ridge()
parameters={'alpha':[1e-3,1e-2,1,5,10,20]}
ridge_regressor= RandomizedSearchCV(ridge,parameters,cv=10,scoring='neg_mean_squared_error')
ridge_regressor.fit(X_train,y_train)


# In[58]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

prediction_r=ridge_regressor.predict(X_test)
print('MSE:',mean_absolute_error(y_test,prediction_r))
print('MSE:',mean_squared_error(y_test,prediction_r))
print('RMSE:',np.sqrt(mean_absolute_error(y_test,prediction_r)))
print(f'r2 score of ridge : {r2_score(y_test,prediction_r)}')


# In[60]:


sns.distplot(y_test-prediction_r)


# In[61]:


joblib.dump(ridge_regressor,'iplmodel_ridge.sav')


# In[ ]:




