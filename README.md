# logistics-exploratory-project
Exploration of logistics company dataset with SQL and Python as a final project for Statisitical Data Management


Disclaimer: 
I no longer have access to the compnay database, I just hope to demonstrate skills learnend in the process.

SUMMARY:
Using a real trucking company's database I explore which variables contribute most to the cost of goods sold (COGS). Also created a train and test data set to implement a small logisitic regression model predicting the COGS. Accuracy of the model was 67%

Established an connection to the database using ODBC and pulled a subset of the database using SQL.

Identified variables important to predicting the COGS using python libraries:

pandas and numpy were used for exploratory analysis
seaborn and matplotlib were used for visual representations
sklearn was used to run logistic regression


Improvements:

Need to clean the data before exploration, looking for duplicates, null values and outliers in columns. 

checking for better predictive model s- not confident logistic model was the best fit for this data set. 

While looking back at the incorrect results it was difficult to check the results since the data tables were manipulated and then split into training datasets.
