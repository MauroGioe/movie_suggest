import streamlit as st
import pandas as pd
import numpy as np
from movies import Recommender





st.title('Movie recomnder')

st.text_input("Insert the name of the movie you want a suggestion for", key="movie_name")
st.write("Movie suggestions")

try:
    movies = Recommender(st.session_state.movie_name)
    for movie in movies["Title"]:
        st.write(movie)
except:
    st.write("There are no movies to suggest for this one")

