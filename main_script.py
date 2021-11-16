import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn import metrics
st.set_option('deprecation.showPyplotGlobalUse', False)
import base64

#1 Importing the data
st.title("GUI for Machine Learning Model")

st.subheader("Please upload the dataset")

def get_dataset():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    else:
        df = 0

    return df


dataset = get_dataset()

st.write(""" ##### Shape of dataset""",dataset.shape)
st.write(""" ##### Column names""",dataset.columns)
st.write(""" ##### Dataset overview""",dataset.head())
st.write(""" ##### Nature of the data""",dataset.describe())
st.write(""" ##### Unique variables in each feature""",dataset.nunique())


#2 Find missing values
st.markdown("### Missing values")
miss_val = st.selectbox("Check for missing values",(" ","Yes","No"))
if miss_val == "Yes":
    features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]
    for feature in features_with_na:
        st.write(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')

#3 Plotting the missing values

plot1 = st.radio("Plot the missing values in all the features",("Yes","No"))
if plot1 == "Yes":
    for feature in features_with_na:
        data = dataset.copy()
        # let's make a variable that indicates 1 if the observation was missing or zero otherwise
        data[feature] = np.where(data[feature].isnull(), 1, 0)

        # let's calculate the mean SalePrice where the information is missing or present
        data.groupby(feature)['SalePrice'].median().plot.bar()
        plt.title(feature)
        plt.show()
        st.pyplot()

#4 Numerical variables
var1 = st.selectbox("Select the type of variable to analyse",(" ","Numerical variables","Categorical variables"))
if var1 == "Numerical variables":

    numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

    st.write('Number of numerical variables: ', len(numerical_features))
    st.write(dataset[numerical_features].head())

#5 Categorical variables
if var1 == "Categorical variables":
    categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']
    st.write("Number of categorical_features",len(categorical_features))
    st.write(dataset[categorical_features].head())

#6 Outlier detection
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
st.markdown("### Outlier detection")
outlier = st.selectbox("Plot out the outliers in the data",(" ","Yes","No"))
if outlier == "Yes":
    for feature in continuous_feature:
        data=dataset.copy()
        if 0 in data[feature].unique():
            pass
        else:
            data[feature]=np.log(data[feature])
            data.boxplot(column=feature)
            plt.ylabel(feature)
            plt.title(feature)
            plt.show()
            st.pyplot()

# 09-11-2021

#1
categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']
st.subheader("Finding the relation between categorical and dependent features")
char = st.selectbox("Plots between categorical and dependent?",(" ","Yes","No"))
if char == "Yes":
    for feature in categorical_features:
        data=dataset.copy()
        data.groupby(feature)['SalePrice'].median().plot.bar()
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()
        st.pyplot()

#2
st.subheader("Feature Engineering")
miss = st.selectbox("Display the missing values",(" ","Yes","No"))
if miss == "Yes":
    features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']
    for feature in features_nan:
        st.write("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))

#3
corr = st.selectbox("Do you want to correct the missing values?",(" ","Yes","No"))
def replace_cat_feature(dataset,features_nan):
    data=dataset.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data
if corr == "Yes":
    dataset=replace_cat_feature(dataset,features_nan)
    st.write(dataset[features_nan].isnull().sum())

#4
dis = st.selectbox("Display missing values in numerical data and correct it?",(" ","Yes","No"))
if dis == "Yes":
    numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']
    for feature in numerical_with_nan:
        st.write("{}: {}% missing value".format(feature,np.around(dataset[feature].isnull().mean(),4)))
    for feature in numerical_with_nan:
        ## We will replace by using median since there are outliers
        median_value=dataset[feature].median()
        ## create a new feature to capture nan values
        dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
        dataset[feature].fillna(median_value,inplace=True)
    st.markdown("### Corrected features")
    st.write(dataset[numerical_with_nan].isnull().sum())

#5
st.markdown("### Display the temporal variables")
temp = st.selectbox("Display the temporal variables",("","Yes","No" ))
if temp == "Yes":
    ## Temporal Variables (Date Time Variables)
    for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
        dataset[feature]=dataset['YrSold']-dataset[feature]
    st.write(dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head())

#6
log = st.selectbox("Apply log transformation",(" ","Yes","No"))
if log == "Yes":
    num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
    for feature in num_features:
        dataset[feature]=np.log(dataset[feature])
    # Temporal Variables (Categorical Features)
    categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']
    for feature in categorical_features:
        temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)
        temp_df=temp[temp>0.01].index
        dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')
    for feature in categorical_features:
        labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
        labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
        dataset[feature]=dataset[feature].map(labels_ordered)
    st.write(dataset.head())

# 16-11-2021

#1 Standardization and Normalization of data
st.markdown("### Data Transformation")
stand = st.selectbox("Select the data transformation technique to be implemented",("Select","Standardization","Normalization"))
scaling_feature=[feature for feature in dataset.columns if feature not in ['Id','SalePerice'] ]
if stand == "Standardization":
    scaler=MinMaxScaler()
    scaler.fit(dataset[scaling_feature])
    scaler.transform(dataset[scaling_feature])
    data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                        pd.DataFrame(scaler.transform(dataset[scaling_feature]), columns=scaling_feature)],
                        axis=1)

if stand == "Normalization":
    scaler=StandardScaler()
    scaler.fit(dataset[scaling_feature])
    scaler.transform(dataset[scaling_feature])
    data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                        pd.DataFrame(scaler.transform(dataset[scaling_feature]), columns=scaling_feature)],
                        axis=1)

#2 Splitting the data
split = st.selectbox("Split the data",("Select","Yes","No"))
if split == "Yes":
    # Splitting the data
    x = data.drop("SalePrice",axis = 1)
    y = data["SalePrice"]
    y.columns = ["Delete","SalePrice"]
    y.drop(["Delete"],axis = 1,inplace = True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 30,test_size = 0.33)
    st.write("x_train shape",x_train.shape)
    st.write("x_test shape",x_test.shape)
    st.write("y_train shape",y_train.shape)
    st.write("y_test shape",y_test.shape)

#3 Linear Regression

st.markdown("### Model Building")
mod = st.selectbox("Select the machine learning model to train on",("Select","Linear Regression","Lasso Regression","K Nearest Neighbours",
                                                                    "Random Forest"))
if mod == "Linear Regression":
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    predictions = lm.predict(x_test)
    plt.scatter(y_test,predictions)
    sns.distplot((y_test-predictions),bins = 50)
    st.pyplot()

#4 Lasso
if mod == "Lasso Regression":
    ridge = Ridge(alpha=.3)
    ridge.fit(x_train,y_train)
    st.write("Ridge Regression Model Training Score: ",ridge.score(x_train, y_train))
    y_pred = ridge.predict(x_test)

#5 K Nearest
if mod == "K Nearest Neighbours":
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(x_train, y_train)
    train_preds = knn_model.predict(x_test)

#6 Random Forest
if mod == "Random Forest":
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(x_train,y_train)
    predictions = rf.predict(x_train)











