import streamlit as st
import pandas as pd

leaderboard = pd.read_csv("leaderboard/leaderboard.csv")

st.title("Food Classification Leaderboard")

st.dataframe(
    leaderboard.sort_values("accuracy", ascending=False)
)