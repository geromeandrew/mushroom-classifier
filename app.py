import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title('Binary Classification Web Application')
    st.sidebar.title('Binary Classification Web Application')
    st.markdown('Are your mushrooms edible?  🍄')
    st.sidebar.markdown('Are your mushrooms edible?  🍄')

    @st.cache_data(persist=True)
    def load_data():
        df = pd.read_csv('mushrooms.csv')
        label_encoder = LabelEncoder()
        for col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])
        return df

    @st.cache_data(persist=True)
    def split_data(dataset):
        X = np.array(dataset.drop('type', axis=1))
        y = np.array(dataset['type'])

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(model, metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            fig, ax = plt.subplots()
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names, ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader('Receiver Operating Characteristic Curve')
            fig, ax = plt.subplots()
            plot_roc_curve(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            fig, ax = plt.subplots()
            plot_precision_recall_curve(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    df = load_data()
    x_train, x_test, y_train, y_test = split_data(df)
    class_names = ['Edible', 'Poisonous']
    metrics_list = ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve']

    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Mushroom dataset for classification')
        st.write(df)

    st.sidebar.subheader('Select a Classifier')
    classifier = st.sidebar.selectbox(
        'Classifier: ', ('Support Vector Machine', 'Logistic Regression', 'Random Forest'))

    if classifier == 'Support Vector Machine':
        st.sidebar.subheader('Model Hyperparameters')
        λ = st.sidebar.number_input(
            'λ (Regularization Parameter): ', 0.01, 10.0, step=0.01, key='λ_svm')
        kernel = st.sidebar.radio('Kernel: ', ('rbf', 'linear'), key='kernel')
        γ = st.sidebar.radio('γ (Kernel Coefficient): ',
                             ('scale', 'auto'), key='γ')

        metrics = st.sidebar.multiselect(
            'Select which metrics to plot: ', metrics_list)

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Support Vector Machine Classification Results: ')
            model = SVC(C=λ, kernel=kernel, gamma=γ)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(
                y_test, predictions, labels=class_names)
            recall = recall_score(y_test, predictions, labels=class_names)
            st.write(f'Accuracy: {accuracy: .2f}')
            st.write(f'Precision: {precision: .2f}')
            st.write(f'Recall: {recall: .2f}')
            plot_metrics(model, metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        λ = st.sidebar.number_input(
            'λ (Regularization Parameter): ', 0.01, 10.0, step=0.01, key='λ_lr')
        max_iter = st.sidebar.slider(
            'Maximum Number of Iterations:', 1, 500, key='max_iter')

        metrics = st.sidebar.multiselect(
            'Select which metrics to plot: ', metrics_list)

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Logistic Regression Classification Results: ')
            model = LogisticRegression(C=λ, max_iter=max_iter)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(
                y_test, predictions, labels=class_names)
            recall = recall_score(y_test, predictions, labels=class_names)
            st.write(f'Accuracy: {accuracy: .2f}')
            st.write(f'Precision: {precision: .2f}')
            st.write(f'Recall: {recall: .2f}')
            plot_metrics(model, metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input(
            'Number of trees in the forest: ', 100, 500, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input(
            'Maximum depth of the forest: ', 1, 20, step=1, key='max_depth')
        metrics = st.sidebar.multiselect(
            'Select which metrics to plot: ', metrics_list)

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Random Forest Classification Results: ')
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = model.score(x_test, y_test)
            precision = precision_score(
                y_test, predictions, labels=class_names)
            recall = recall_score(y_test, predictions, labels=class_names)
            st.write(f'Accuracy: {accuracy: .2f}')
            st.write(f'Precision: {precision: .2f}')
            st.write(f'Recall: {recall: .2f}')
            plot_metrics(model, metrics)


if __name__ == '__main__':
    main()
