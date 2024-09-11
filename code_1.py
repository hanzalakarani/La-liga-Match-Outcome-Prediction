import pandas as pd
import joblib

# Load the dataset
file_path = "C:/Users/USER/Desktop/laliga/new_laliga.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset and its structure
data.head()
data.info()
data.shape
data['FTR'].value_counts()
data.describe()

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'],format='mixed' )

# Check for missing values
missing_values = data.isnull().sum()
res = pd.get_dummies(data, columns=['HomeTeam', 'AwayTeam'])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['HS', 'AS', 'HST', 'AST']] = scaler.fit_transform(data[['HS', 'AS', 'HST', 'AST']])
from sklearn.model_selection import train_test_split
# Drop features that cause data leakage
X = data.drop(labels=['FTR', 'Div', 'Date', 'HomeTeam', 'AwayTeam', 'HTR', 
                      'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
                      'HomePossessionProxy', 'AwayPossessionProxy', 'Outcome', 
                      'HTHG', 'HTAG', 'FTHG', 'FTAG', 'GoalDifference'], axis=1)
y = data['FTR']

# Remove non-numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=[float, int])
correlation_matrix = numeric_data.corr()

# Display correlation with 'FTR'
print(correlation_matrix['total_goals'].sort_values(ascending=False))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: ", best_params)
print("Best Cross-Validation Score: ", best_score)

# Use the best estimator to make predictions
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# Evaluate the optimized model
print("Optimized Random Forest Accuracy: ", accuracy_score(y_test, y_pred_best_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_best_rf))



joblib.dump(best_rf, 'C:/Users/USER/Desktop/laliga/model.joblib')



