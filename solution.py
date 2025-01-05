import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel('data_SEN.xlsx', engine='openpyxl')
def binning(target,bins):
    remainder=0
    for i in range(len(target)):
        timestamp = target.index[i]
        value = target.iloc[i] 
        minval=500000 # +infinity
        for j in bins:
            if abs(value-j+remainder) < minval:
                optval = j
                minval = abs(value-j+remainder)
                nextremainder = value+remainder-j
        target.iloc[i] = optval
        remainder=nextremainder
    return target
def XLabels(target):
    year = target.index.year
    yearEncoder=LabelEncoder()
    year = yearEncoder.fit_transform(year)
    month = target.index.month
    monthEncoder = LabelEncoder()
    month = monthEncoder.fit_transform(month)
    days_of_week = target.index.day_name()
    dayEncoder = LabelEncoder()
    days_of_week = dayEncoder.fit_transform(days_of_week)
    hours = target.index.hour
    hourEncoder = LabelEncoder()
    hours = hourEncoder.fit_transform(hours)
    prevVal = [0]+target.tolist()[:-1]
    prevEncoder = LabelEncoder()
    prevVal = prevEncoder.fit_transform(prevVal)
    prevVal = [0]+target.tolist()[:-1]
    return (year,month,days_of_week,hours)
def pred(column_name,hourly_data_train,hourly_data_test,bins=[-500,-300,-100,0,100,300,500]):
    target=hourly_data_train[column_name].diff()
    target.dropna(inplace=True)
    targetTest=hourly_data_test[column_name].diff()
    targetTest.dropna(inplace=True)
    clf = DecisionTreeClassifier(criterion='entropy')

    target=binning(target,bins)
    (year,month,days_of_week,hours)=XLabels(target)
    (Tyear,Tmonth,Tdays_of_week,Thours)=XLabels(targetTest)
    X_train=np.array([year,month,days_of_week,hours]).T
    X_test= np.array([Tyear,Tmonth,Tdays_of_week,Thours]).T
    clf.fit(X_train, target.to_numpy())
    clfNB= CategoricalNB()
    clfNB.fit(X_train,target.to_numpy())
    predictions = clf.predict(X_test)
    predictionsNB= clfNB.predict(X_test)
    (ID3Error,actualList,predList) =totalErrorMAE(predictions,targetTest)
    plt.figure(figsize=(8, 5))
    plt.plot(actualList, label='Actual', marker='o')
    plt.plot(predList, label='Predicted', marker='x')
    plt.xlabel('December')
    plt.ylabel(column_name)
    plt.title('Prediction vs Actual ID3')
    plt.legend()
    plt.grid(True)
    plt.show()
    (NBError,actualList,predList) = totalErrorMAE(predictionsNB,targetTest)
    plt.figure(figsize=(8, 5))
    plt.plot(actualList, label='Actual', marker='o')
    plt.plot(predList, label='Predicted', marker='x')
    plt.xlabel('December')
    plt.ylabel(column_name)
    plt.title('Prediction vs Actual Naive Bayes')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f"{column_name} MAE ID3 : {ID3Error}")
    print(f"{column_name} MAE Naive Bayes : {NBError}")
def totalErrorMAE(predictions,targetTest):
    totalError=0
    totalTest=0
    totalPred=0
    predList=[]
    actualList=[]
    for i in range(len(predictions)):
        totalTest+=targetTest.iloc[i]
        totalPred+=predictions[i]
        actualList+=[totalTest]
        predList+=[totalPred]
        totalError+=abs(totalTest-totalPred)
    return (totalError/len(predictions),actualList,predList)

df['Data'] = pd.to_datetime(df['Data'],dayfirst=True)
df.set_index('Data', inplace=True)
print(df.dtypes)
# convert type to numeric int64 and delete NaN rows 
for column in df.iloc[:, 1:]:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=[column], inplace=True)
    df[column]=df[column].astype('int64')
print(df.dtypes)
print(df.head)
hourly_data = df.select_dtypes(include=['number']).resample('H').mean()
print(hourly_data.head)
daily_df = hourly_data.resample('D').mean()

daily_df.reset_index(inplace=True)
daily_df.set_index('Data', inplace=True)
weekly_df = daily_df.resample('W').mean()
weekly_df.reset_index(inplace=True)
hourly_data.reset_index(inplace=True)
hourly_data.set_index('Data', inplace=True)
print(daily_df.head)
# Date when test set begins and train set ends
split_date = pd.to_datetime('2024-12-01')
hourly_data_train = hourly_data[hourly_data.index < split_date]
hourly_data_test = hourly_data[hourly_data.index >= split_date]
#pred("Medie Consum[MW]",hourly_data_train,hourly_data_test)
#pred("Productie[MW]",hourly_data_train,hourly_data_test,[-300,-100,0,100,300])
pred("Sold[MW]",hourly_data_train,hourly_data_test)
