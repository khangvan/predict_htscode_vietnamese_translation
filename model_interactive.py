import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy
import pandas
st.title('Khang 2')

st.write("tesst Here's our first attempt at using data to create a table:")
st.write(pandas.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

txt = st.text_area('Text to analyze', '''
 It was the best of times, it was the worst of times, it was
 the age of wisdom, it was the age of foolishness, it was
 the epoch of belief, it was the epoch of incredulity, it
 was the season of Light, it was the season of Darkness, it
 was the spring of hope, it was the winter of despair, (...)
 ''')

st.write('Sentiment:', (txt))