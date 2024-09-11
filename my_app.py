import streamlit as st
import pandas as pd
import joblib

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
        














        
