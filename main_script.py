import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
st.set_option('deprecation.showPyplotGlobalUse', False)
import base64

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


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

plot1 = st.radio("Plot the missing values in all the features",(" ","Yes","No"))
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

