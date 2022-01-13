import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

X_train = ''
X_test = ''
y_train = ''
y_test  = ''

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache  
def app():
    
    st.header("Machine Learning Algorithm Comparison")
st.header("""
          Machine Learning Algorithm Comparison
         Explore different classifiers
         Which one is the have the highest accuracy?
         """ )
data = st.file_uploader("Upload dataset:",type=['csv','xlsx'])
st.success("Data successfully loaded")

if data is not None:
    df=pd.read_csv(data,';')
    st.dataframe(df)
    
    le_CreditHistory = LabelEncoder()
    df['CreditHistory'] = le_CreditHistory.fit_transform(df['CreditHistory'])
    df["CreditHistory"].unique()
    
    le_Employment = LabelEncoder()
    df['Employment'] = le_Employment.fit_transform(df['Employment'])
    df["Employment"].unique()
  
 
    if st.checkbox('Select Multiple Columns'):
        new_data = st.multiselect('Select preferred colunmn features',df.columns)
        df1=df[new_data]
        st.dataframe(df1)
        
        X = df1.iloc[:,0:-1]
        y=df1.iloc[:,-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

        #Creating Models
clf = LazyClassifier(ignore_warnings=True, custom_metric=None)
if X_train is not '':
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    st.write('Performance of the Models',models)


if st.checkbox('Bar Chart of the models'):
     plt.figure(figsize=(5,10))
     sns.set_theme(style="darkgrid")
     ax = sns.barplot(y=models.index,x="Accuracy",data=models)
     st.pyplot()
     
     
if st.checkbox("Display shape"):
        st.write(df.shape)
if st.checkbox("Display Count 0 = Bad & 1 = Good"):
        st.write(df.Default.value_counts())
#if st.checkbox("Display Correlation of data"):
        #st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
        #st.pyplot()
        
# Sidebar - Specify parameter settings
with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

