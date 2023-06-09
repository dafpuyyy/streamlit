import streamlit as st
import pandas as pd
from statistics import NormalDist
from statsmodels.stats.weightstats import ztest
import numpy as np
from scipy.stats import ttest_1samp, t
from scipy.stats import f_oneway
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns 

with st.sidebar:
    tipe = st.radio('Pilih Menu', ['Z-test' ,'Z-test(manual)', 'T-test','T-test(manual)'])

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if tipe == 'Z-test':
    st.title('Z-test')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("View Data"):
            st.dataframe(df.transpose())

        with st.expander("View Statistics"):
            st.dataframe(df.describe().transpose())
        st.write('## Construction of Hypothesis')
        alpha = 0.05
        alpha_z = NormalDist().inv_cdf(p=1 - alpha / 2)

        selected_column = st.selectbox("Select Column", df.columns)
        column_data = df[selected_column]

        null_mean = st.number_input("Enter Null Mean", value=0.0)

        z_score, p_value = ztest(column_data, value=null_mean, alternative='two-sided')

        clicked = st.button('Do the z-test!')

        if clicked:
            if abs(z_score) > alpha_z:
                st.write("Reject H0")
            else:
                st.write("Cannot reject H0")
            st.write(z_score, alpha_z)
            fig, ax = plt.subplots()
            x = np.linspace(-5, 5, 100)
            y = norm.pdf(x, loc=0, scale=1)
            ax.plot(x, y, label='Standard Normal Distribution')
            ax.fill_between(x[x >= abs(z_score)], y[x >= abs(z_score)], alpha=0.5, label='Rejection Region')
            ax.axvline(z_score, color='red', linestyle='--', label='z-score')
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("Please upload a CSV file.")

elif tipe == 'Z-test(manual)':
    st.title('Z-test')

    xbar = st.number_input("Sample Mean (xbar)", value=0.0)
    mu = st.number_input("Population Mean (mu)", value=0.0)
    stdev = st.number_input("Population Standard Deviation (stdev)", value=1.0)
    n = st.number_input("Sample Size (n)", value=1)
    alpha = 0.05
    z_score = (xbar - mu) / (stdev / (n ** 0.5))

    clicked = st.button('Do the Z-test!')

    if clicked:
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        if p_value < alpha:
            st.write("Reject H0")
        else:
            st.write("Cannot reject H0")

        st.write("Z-score:", z_score)
        st.write("p-value:", p_value)
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x, loc=0, scale=1)

        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.fill_between(x, 0, y, where=(x >= -z_score) & (x <= z_score), color='blue', alpha=0.3)
        ax.fill_between(x, 0, y, where=(x < -z_score) | (x > z_score), color='gray', alpha=0.3)

        ax.axvline(x=z_score, color='red', linestyle='--')
        ax.axvline(x=-z_score, color='red', linestyle='--')

        ax.set_title('Z-test Distribution')
        ax.set_xlabel('Z-score')
        ax.set_ylabel('Probability Density')
        st.pyplot(fig)
elif tipe == 'T-test':
    st.title('T-test')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("View Data"):
            st.dataframe(df.transpose())

        with st.expander("View Statistics"):
            st.dataframe(df.describe().transpose())

        st.write('## Construction of Hypothesis')
        alpha = 0.05

        selected_column = st.selectbox("Select Column", df.columns)
        column_data = df[selected_column]

        x_bar = column_data.mean()
        sigma = column_data.std()
        n = len(column_data)

        null_mean = st.number_input("Enter Null Mean", value=0.0)

        t_value, p_value = ttest_1samp(column_data, popmean=null_mean)

        clicked = st.button('Do the t-test!')

        if clicked:
            if abs(t_value) > t.ppf(1 - alpha / 2, df=n-1):
                st.write("Reject H0")
            else:
                st.write("Cannot reject H0")

            fig, ax = plt.subplots()
            x = np.linspace(-5, 5, 100)
            y = t.pdf(x, df=n-1)
            ax.plot(x, y, label='t-Distribution (df={})'.format(n-1))
            ax.fill_between(x[x >= abs(t_value)], y[x >= abs(t_value)], alpha=0.5, label='Rejection Region')
            ax.axvline(t_value, color='red', linestyle='--', label='t-value')
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("Please upload a CSV file.")
elif tipe == 'F-test':
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        with st.expander("View Data"):
            st.dataframe(df.transpose())

        with st.expander("View Statistics"):
            st.dataframe(df.describe().transpose())

        st.write('## Construction of Hypothesis')
        alpha = 0.05

        selected_columns = st.multiselect("Select Columns", df.columns)

        if len(selected_columns) == 2:
            column_data1 = df[selected_columns[0]]
            column_data2 = df[selected_columns[1]]

            f_value, p_value = f_oneway(column_data1, column_data2)

            clicked = st.button('Do the F-test!')

            if clicked:
                if p_value < alpha:
                    st.write("Reject H0")
                else:
                    st.write("Cannot reject H0")

            st.write("F-value:", f_value)
            st.write("p-value:", p_value)

    else:
        st.write("Please upload a CSV file.")
elif tipe == 'T-test(manual)':
    xbar = st.number_input("Sample Mean (xbar)", value=0.0)
    mu = st.number_input("Population Mean (mu)", value=0.0)
    s = st.number_input("Sample Standard Deviation (s)", value=1.0)
    n = st.number_input("Sample Size (n)", value=1, min_value=1)

    alpha = 0.05

    t_score = (xbar - mu) / (s / (n ** 0.5))

    clicked = st.button('Do the T-test!')

    if clicked:
        p_value = 2 * (1 - t.cdf(abs(t_score), df=n-1))

        if p_value < alpha:
            st.write("Reject H0")
        else:
            st.write("Cannot reject H0")

        st.write("T-score:", t_score)
        st.write("p-value:", p_value)
        x = np.linspace(-4, 4, 100)
        y = t.pdf(x, df=n-1)

        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.fill_between(x, 0, y, where=(x >= -t_score) & (x <= t_score), color='blue', alpha=0.3)
        ax.fill_between(x, 0, y, where=(x < -t_score) | (x > t_score), color='gray', alpha=0.3)

        ax.axvline(x=t_score, color='red', linestyle='--')
        ax.axvline(x=-t_score, color='red', linestyle='--')

        ax.set_title('T-test Distribution')
        ax.set_xlabel('T-score')
        ax.set_ylabel('Probability Density')

        st.pyplot(fig)