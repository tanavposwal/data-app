import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Football Explorer")

st.markdown("""
This app performs a simple webscrapping of NFL football player stats data:
* **Python libraries:** base64, pandas, seaborn
* **Data source:** https://www.pro-football-reference.com/
""")

st.sidebar.header("User Input Features")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1990, 2020))))


@st.cache_data
def load_data(year):
    url = f"https://www.pro-football-reference.com/years/{str(year)}/rushing.htm"
    html = pd.read_html(url, header=1)
    df = html[0]
    raw = df.drop(df[df.Age == "Age"].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(["Rk"], axis=1)
    return playerstats


playerstats = load_data(selected_year)

# sidebar - team selection
sorted_unique_team = playerstats.Team.unique()
selected_team = st.sidebar.multiselect("Team", sorted_unique_team, sorted_unique_team)

unique_pos = ["RB", "QB", "WR", "FB", "TE"]
selected_pos = st.sidebar.multiselect("Position", unique_pos, unique_pos)

df_selected_team = playerstats[
    (playerstats.Team.isin(selected_team) & (playerstats.Pos.isin(selected_pos)))
]

st.header("Display Player Stats of Selected Team(s)")
st.write(
    "Data dimensions:"
    + str(df_selected_team.shape[0])
    + " rows x "
    + str(df_selected_team.shape[1])
    + " columns"
)
st.dataframe(df_selected_team)


def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv</a>'
    return href


st.markdown(file_download(df_selected_team), unsafe_allow_html=True)

# heatmap
if st.button("Intercorrelation Heatmap"):
    st.header("Intercorrelation Heatmap Matrix")
    df_selected_team.to_csv("output.csv", index=False)
    df = pd.read_csv("output.csv")
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
