import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#external file that contains connection to database
from Py_Sql_Alchemy_Class_DePaul import DB_Table_Ops

import sklearn.linear_model as sk
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


dbto = DB_Table_Ops()

sql_comm1 = """SELECT l.LoadGuId,l.COGS,l.Mode,MONTH(l.CreatedDate) as CreatedMonth,l.Distance, 
                lf.Weight,lf.Quantity, lf.HandlingUnit, lf.Class, 
                e.Category
               FROM Loads as l
               TABLESAMPLE(5 PERCENT)
               INNER JOIN LoadFreight as lf
               ON l.LoadGuId = lf.LoadGuId
               INNER JOIN LoadEquipment as le
               ON l.LoadGuId = le.LoadGuID
               INNER JOIN Equipment as e
               ON le.EquipmentId = e.EquipmentID
               WHERE l.COGS > 0
               AND l.Distance > 0
               """
df = dbto.query_to_df(sql_comm1, index_col='LoadGuId')
df.head()

#Exploratory Data Analysis-------------------------------------
df.describe()
#Understanding what factors are involved in calculating COGS
#Assumptions Primary Factors include Distance, Mode, Weight, Quantity, Equipment Category, and CreatedDate


#Correlation table to see what numerical columns are related
df.corr()
#Results - Distance and weight seem to be the most impactful among numerical columns


#Exploring CreatedDate - Does when the item is being shipped play into COGS, 
#Do winter months with more shipping (holiday season) increase cost?

cd = df.groupby('CreatedMonth')['COGS'].mean()
cd.plot.bar(title = "COGS Per Month",ylabel='COGS')

#Data shows a slight decrease in COGS for the first 2 months but no real change in the others

#Exploring Equipment Category

e = df.groupby('Category')['COGS'].mean()
e.plot.bar(title = "COGS By Category",ylabel='COGS')

#Category likely has an impact Containers/Reefer Category have higher COGS on average than Vans

#Exploring Mode
m = df.groupby('Mode')['COGS'].mean()
m.plot.bar(title = "COGS By Mode",ylabel='COGS')

#Mode Likely has an impact with LTL being the least expensive and Intermodal and International the most

#Exploring HandlingUnit

hu = df.groupby('HandlingUnit')['COGS'].mean()
hu.plot.bar(title = "COGS By HandlingUnit",ylabel='COGS')

#bales semmingly cost the most with corrugated being the least on average

#Catplot for mode and cogs grouped by category
#fix axis

cp= sns.catplot(data=df, x="Mode", y="COGS", hue="Category", kind="bar").set(title='Average COGS per Mode grouped by Category')
for ax in cp.axes.ravel():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
cp

#Created a filtered Data table for more detailed plots,
#removing outliers using upper and lower quantiles

q_low = df["COGS"].quantile(0.05)
q_hi  = df["COGS"].quantile(0.95)
df_filtered = df[(df["COGS"] < q_hi) & (df["COGS"] > q_low)]
q_l = df_filtered["Distance"].quantile(0.10)
q_h  = df_filtered["Distance"].quantile(0.90)
df_filtered = df_filtered[(df_filtered["Distance"] < q_h) & (df_filtered["Distance"] > q_l)]

sp=sns.scatterplot(data = df_filtered, x = 'Distance', y = 'COGS',hue='Category').set(title='Distance vs COGS grouped by Category')

sp

#Removing Van from observation just to see the other categories

df_filtered = df_filtered[df_filtered.Category != 'Van']
sp2=sns.scatterplot(data = df_filtered, x = 'Distance', y = 'COGS',hue='Category', palette='Set2').set(title='Distance vs COGS grouped by Category')
sp2


#similar scatterplot with Mode being the category
s=sns.scatterplot(data = df_filtered, x = 'Distance', y = 'COGS',hue='Mode')
print(s)

#as distance increases so do most of the modes cogs, LTL-increase the least (from earlier we know most LTL are on vans)

q_low = df_filtered["Weight"].quantile(0.05)
q_hi  = df_filtered["Weight"].quantile(0.95)
df_filtered = df_filtered[(df_filtered["Weight"] < q_hi) & (df_filtered["Weight"] > q_low)]

hp=sns.histplot(data=df_filtered, x="Weight", hue="HandlingUnit", multiple="stack")
print(hp)

#Predictive Model---------------------------------------------------------------


#Hot coding categorical data into 1s and 0s for better classification
df_dc = pd.get_dummies(df, columns=['Category','HandlingUnit','Mode'])
df_dc.head()

#Grouping COGS into bins so values of test can fall into ranges
df_dc['COGS Groups'] = pd.qcut(df_dc['COGS'],[0,0.2,0.4,0.6,0.8, 1])
df_dc['COGS Groups'] = df_dc['COGS Groups'].astype(str)


#Choosing what data to include in the regression model
X= df_dc[['Quantity','Distance','Weight','Category_Container','Category_Flatbed / Deck','Category_Reefer','Category_Specialized','Category_Van',
            'HandlingUnit_Gaylords','HandlingUnit_Pallets','HandlingUnit_Rolls','HandlingUnit_Skids','HandlingUnit_Bales','HandlingUnit_Drums','HandlingUnit_Floor','HandlingUnit_Corrugated','HandlingUnit_Crates',
            'Mode_Expedited','Mode_Intermodal','Mode_International - Ocean','Mode_LTL','Mode_Partial Load','Mode_TL']]
Y= df_dc[['COGS Groups']] 

#Splitting the data into train and test data
X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=0.3)

#running Logistic Regression for classification
logreg= sk.LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print(classification_report(y_test, y_pred))


matrix = plot_confusion_matrix(logreg,X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation = 90)
plt.show()



#Exploring incorrect results to learn if any factor led to more errors than the other
#scatterplot where distance was not right
#every incorrect bucket show the distance

temp1=X_test
temp1['Predictions'] = y_pred.tolist()
temp1 = temp1.join(y_test['COGS Groups'])
temp1['Correct/Incorrect'] =np.where(temp1['Predictions'] == temp1['COGS Groups'],1,0)

plt.bar(x = 'COGS Groups', height = 'Distance', data = temp1[temp1['Correct/Incorrect']==0])
plt.title('Incorrect Groups by average Distance')
plt.xlabel("COGS Groups")
plt.ylabel("Average Distance")
plt.xticks(rotation = 90)

s=sns.scatterplot(data = temp1[temp1['Correct/Incorrect']==0], x = 'Weight', y = 'COGS Groups',palette='Set2').set(title='Weight vs COGS')
s


#Weaknesses------------------------------------------

#confidence in TABLESAMPLE - is TABELSAMPLE A Truly random sample of the table
#including other tables - CustomerBilling/CarrierBilling - Amounts may have a relation to cogs that should be explored
#bucketing of COGS - included to few buckets in the groupings, which inflates the accuracy of the model
#Not recoding categorical data back into results data table, this would have made for more visually appealing data charts
