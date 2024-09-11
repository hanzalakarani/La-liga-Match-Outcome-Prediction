import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model
model = joblib.load('C:/Users/USER/Desktop/laliga/model.joblib')

# Load the dataset
team_stats = pd.read_csv('C:/Users/USER/Desktop/laliga/new_laliga.csv')

# Dropping unnecessary columns
team_stats = team_stats.drop(labels=['FTR', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 'HTR', 
                                     'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
                                     'HomePossessionProxy', 'AwayPossessionProxy', 'Outcome', 
                                     'HTHG', 'HTAG', 'FTHG', 'FTAG', 'GoalDifference'], axis=1)

# Create a dictionary for team codes and names (Example mapping)
team_code_mapping = {
    0: 'Alaves',
    1:'Almeria',
    2:'Ath Bilbao',
    3:'Ath Madrid', 
    4: 'Barcelona',
    5:'Betis',
    6:'Cadiz',
    7:'Celta',
    8:'Cordoba',
    9:'Eibar',
    10:'Elche',
    11:'Espanol',
    12:'Getafe',
    13:'Gimnastic',
    14:'Girona',
    15:'Granada',
    16:'Hercules',
    17:'Huesca',
    18:'La Coruna',
    19:'Las Palmas',
    20:'Leganes',
    21:'levante',
    22:'Malaga',
    23:'Mallorca',
    24:'Murcia',
    25:'Numancia',
    26:'Osasuna',
    27: 'Real Madrid',
    28:'Recreativo',
    29:'Santander',
    30:'Sevilla',
    31:'Sociedad',
    32:'Sp Gijon',
    33:'Tenerife',
    34:'Valencia',
    35:'Valladolid',
    36:'Vallecano',
    37:'Villarreal',
    38:'Xerez',
    39:'Zaragoza'
}

# Adding the new mapping as a DataFrame
team_codes_df = pd.DataFrame(list(team_code_mapping.items()), columns=['Team Code', 'Team Name'])

# Set up the dashboard layout
st.title("La Liga Match Outcome Prediction Dashboard")

# **New Section: List of Teams with Codes and Names**
#st.subheader("List of Teams with Team Codes and Names")
#st.table(team_codes_df)  # Displaying the list of teams with their codes and names

# Sidebar for team statistics
st.sidebar.title("Team Statistics & Insights")
selected_team = st.sidebar.selectbox("Select a team", team_codes_df['Team Name'])

# Use the selected team to retrieve corresponding code for prediction or display
team_code = team_codes_df[team_codes_df['Team Name'] == selected_team]['Team Code'].values[0]
team_data = team_stats[team_stats['HT_Code'] == team_code].mean()

st.sidebar.write(f"Average Statistics for {selected_team}:")
st.sidebar.write(team_data)

# Additional sidebar visuals: Team Shot Efficiency
st.sidebar.subheader("Shot Efficiency Comparison")
fig = go.Figure(data=[
    go.Bar(name='Home Shot Efficiency', x=[selected_team], y=[team_data['HomeShotEfficiency']]),
    go.Bar(name='Away Shot Efficiency', x=[selected_team], y=[team_data['AwayShotEfficiency']])
])
fig.update_layout(barmode='group', title="Home vs Away Shot Efficiency")
st.sidebar.plotly_chart(fig)

# Dropdowns for predicting match outcome
st.header("Predict Match Outcome")
home_team = st.selectbox("Select Home Team", team_codes_df['Team Name'])
away_team = st.selectbox("Select Away Team", team_codes_df['Team Name'])
if home_team==away_team:
    st.write(f"PLEASE SELECT TWO SEPARATE TEAMS.")


# Get the codes of selected home and away teams
home_team_code = team_codes_df[team_codes_df['Team Name'] == home_team]['Team Code'].values[0]
away_team_code = team_codes_df[team_codes_df['Team Name'] == away_team]['Team Code'].values[0]

# Prediction button
if st.button("Predict Outcome"):
    # Get stats for home and away teams
    home_team_stats = team_stats[team_stats['HT_Code'] == home_team_code].mean()
    away_team_stats = team_stats[team_stats['AT_Code'] == away_team_code].mean()
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'HS': [home_team_stats['HS']],
        'AS': [away_team_stats['AS']],
        'HST': [home_team_stats['HST']],
        'AST': [away_team_stats['AST']],
        'HT_Code': [home_team_stats['HT_Code']],
        'AT_Code': [away_team_stats['AT_Code']],
        'HomeShotEfficiency': [home_team_stats['HomeShotEfficiency']],
        'AwayShotEfficiency': [away_team_stats['AwayShotEfficiency']],
        'total_goals': [home_team_stats['total_goals'] + away_team_stats['total_goals']]
    })
    
    # Make prediction
    result = model.predict(input_data)
    
    # Display the prediction result
    st.write(f"The predicted outcome is: {result[0]}")
    if result[0] == 'H':
        st.write(f"Home Team Wins")
    elif result[0] == 'A':
        st.write(f"Away Team Wins")
    else:
        st.write(f"It's a Draw")
        
# Historical match outcomes
st.header("Historical Match Outcomes")
outcome_chart = px.histogram(team_stats, x='HT_Code', color='HT_Code', barmode='group')
st.plotly_chart(outcome_chart)

# Feature Correlation Heatmap
st.header("Feature Correlation with Match Outcome")
correlation_matrix = team_stats.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# New: Goal Distribution Visualization
st.header("Goal Distribution")
goal_distribution = px.box(team_stats, y='total_goals', color='HT_Code', title="Distribution of Goals Scored by Teams")
st.plotly_chart(goal_distribution)

# New: Team Shot Efficiency Scatter Plot
st.header("Team Shot Efficiency Comparison")
shot_efficiency_scatter = px.scatter(team_stats, x='HomeShotEfficiency', y='AwayShotEfficiency', color='HT_Code',
                                     title="Home vs Away Shot Efficiency by Team")
st.plotly_chart(shot_efficiency_scatter)

# New: Home vs Away Goals Scored
st.header("Home vs Away Goals Scored")
goals_comparison = px.bar(team_stats, x='HT_Code', y=['HS', 'AS'], barmode='group',
                          title="Home vs Away Goals Scored by Team")
st.plotly_chart(goals_comparison)
