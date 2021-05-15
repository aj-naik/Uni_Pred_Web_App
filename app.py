import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Graduate Program Admission Predictor")
    st.sidebar.title("Graduate Program Admission Predictor")
    st.markdown("ML Web App for University Admission Predcition for North American Universities")
    st.markdown("Disclaimer: This web app doesn't give definite chance of admission. It should be used to get a general overview of a user's chance of getting an admit and where the user can improve their chances")
    st.sidebar.markdown("ML Web App for University Admission Predcition for North American Universities")
    
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('data.csv')
        data = data.drop('Serial No.', axis=1)
        return data

    @st.cache(persist=True)
    def split(df):
        y = df['Chance of Admit']
        x = df.drop(columns=['Chance of Admit'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def viz_data(plot_list):
        if 'Correlation Matrix' in plot_list:
            st.subheader("Correlation Matrix")
            corr_matrix = df.corr()
            plt.figure(figsize = (12, 12))
            st.write(sns.heatmap(corr_matrix, annot = True))
            st.pyplot()
            
        if 'Histogram' in plot_list:
            st.subheader("Histogram")
            fig, ax = plt.subplots()
            ax = sns.histplot(df,bins=10)
            st.pyplot(fig)

        if 'Pairplot' in plot_list:
            st.subheader("Pairplot")
            fig = sns.pairplot(df)
            st.pyplot(fig)

    df = load_data()

    x_train, x_test, y_train, y_test = split(df)

    if st.sidebar.checkbox("Predictor",False):
        st.subheader("Enter Data For Prediction")
        gre = st.number_input('GRE Score')
        toefl = st.number_input('TOEFL Score')
        rating = st.number_input('Target University Rating (0-5)')
        sop = st.number_input('Satement Of Purpose (0-5)')
        lor = st.number_input('Letter Of Recomendation (0-5)')
        cgpa = st.number_input('CGPA')
        research = st.number_input('Research Published')

        classify = st.sidebar.selectbox("Load Model", ("Linear Regression","Support Vector Regressor (SVR)","K-Nearest Regressor","Decision Tree", "Random Forest", "XGBoost"))
        
        def predictor_select():
            if classify == 'Linear Regression':
                load_weights = open('LR.pkl', 'rb') 
                classifier = pickle.load(load_weights)
            elif classify == 'Support Vector Regressor (SVR)':
                load_weights = open('SVM.pkl', 'rb') 
                classifier = pickle.load(load_weights)
            elif classify == 'K-Nearest Regressor':
                load_weights = open('KNR.pkl', 'rb')
                classifier = pickle.load(load_weights)
            elif classify == 'Decision Tree':
                load_weights = open('DT.pkl', 'rb') 
                classifier = pickle.load(load_weights)
            elif classify == 'Random Forest':
                load_weights = open('RF.pkl', 'rb') 
                classifier = pickle.load(load_weights)
            elif classify == 'XGBoost':
                load_weights = open('XGB.pkl', 'rb') 
                classifier = pickle.load(load_weights)
            return classifier
        
        if st.button("Predict"): 
            classifier = predictor_select()
            result = classifier.predict([[gre, toefl, rating, sop, lor, cgpa, research]])
            # st.write(result)
            st.success('Your chance of admission is {} %'.format(result[0].round(5)*100))

    if st.sidebar.checkbox("Data Plots", False):
       plot_list = st.sidebar.multiselect("Choose Plots", ('Correlation Matrix', 'Histogram', 'Pairplot'))
       viz_data(plot_list)
        
    if st.sidebar.checkbox("Create a custom model",False):
        st.sidebar.subheader("Choose Model")
        classifier = st.sidebar.selectbox("Model", ("Linear Regression","Support Vector Regressor (SVR)","K-Nearest Regressor","Decision Tree", "Random Forest", "XGBoost"))
        # scaler = st.sidebar.selectbox("Choose Scaler", ("Standard Scaler","Min-Max Scaler","No Scaler"))

        if classifier == 'Linear Regression':
            if st.sidebar.button("Train", key='classify'):
                st.subheader("Linear Regression Results")
                model = LinearRegression()
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write('Model Accuracy is {} percent'.format(accuracy.round(3)*100))
                # st.write('Model Pred is {} percent'.format(y_pred.round(3)*100))
            
            if st.button("Save Model"):
                model = LinearRegression()
                model.fit(x_train, y_train)
                weights =open("LR.pkl",mode = "wb")
                pickle.dump(model,weights)
                weights.close()
                st.success('Your model saved sucessfully')

        if classifier == 'Support Vector Regressor (SVR)':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
            kernel = st.sidebar.radio("Kernel", ('linear', 'poly', 'rbf', 'sigmoid'), key='kernel')
            gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
            
            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Support Vector Regressor (SVR) Results")
                model = SVR(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))

            if st.button("Save Model"):
                model = SVR(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                weights =open("SVM.pkl",mode = "wb")
                pickle.dump(model,weights)
                weights.close()                
                st.success('Your model saved sucessfully')

        if classifier == 'K-Nearest Regressor':
            st.sidebar.subheader("Model Hyperparameters")
            neigh = st.sidebar.number_input("No. of Neighbours")
            algo = st.sidebar.radio("Algorithm", ('auto', 'ball_tree', 'kd_tree', 'brute'), key='algo')
        
            if st.sidebar.button("Predict", key='classify'):
                st.subheader("K-Nearest Regressor Results")
                model = KNeighborsRegressor(n_neighbors = int(neigh), algorithm=algo)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2)*100)

            if st.button("Save Model"):
                model = KNeighborsRegressor(n_neighbors = int(neigh), algorithm=algo)
                model.fit(x_train, y_train)
                weights =open("KNR.pkl",mode = "wb")
                pickle.dump(model,weights)
                weights.close()                
                st.success('Your model saved sucessfully')

        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
            bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))

            if st.button("Save Model"):   
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                model.fit(x_train, y_train)
                weights =open("RF.pkl",mode = "wb")
                pickle.dump(model,weights)
                weights.close()
                st.success('Your model saved sucessfully')

        if classifier == 'Decision Tree':
            st.sidebar.subheader("Model Hyperparameters")
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 30, step=1, key='n_estimators')

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Decision Tree Results")
                model = DecisionTreeRegressor(max_depth=max_depth)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)              
                st.write("Accuracy: ", accuracy.round(2))

            if st.button("Save Model"):
                model = DecisionTreeRegressor(max_depth=max_depth)
                model.fit(x_train, y_train)
                weights =open("DT.pkl",mode = "wb")
                pickle.dump(model,weights)
                weights.close()
                st.success('Your model saved sucessfully')

        if classifier == 'XGBoost':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 30, step=1, key='max_depth')
            C = st.sidebar.number_input("C (Learning Rate parameter)", 0.01, 10.0, step=0.01, key='C_LR')

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("XGBoost Results")
                model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = C,
                    max_depth = max_depth, n_estimators = n_estimators)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)             
                st.write("Accuracy: ", accuracy.round(2))

            if st.button("Save Model"):
                model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = C,
                    max_depth = max_depth, n_estimators = n_estimators)
                model.fit(x_train, y_train)
                weights =open("XGB.pkl",mode = "wb")
                pickle.dump(model,weights)
                weights.close()
                st.success('Your model saved sucessfully')

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Admission Data Set")
        st.write(df)


if __name__ == '__main__':
    main()
