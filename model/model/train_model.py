import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
data = pd.read_csv('model/dataset.csv')

# Feature Selection
X = data[['Platform', 'Genre', 'Release_Year', 'Global_Sales', 'NA_Sales', 'EU_Sales']]
y = data['Price']

# Preprocessing: Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('platform_genre', OneHotEncoder(), ['Platform', 'Genre'])
    ],
    remainder='passthrough'
)

# Build the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
pickle.dump(pipeline, open('model/trained_model.pkl', 'wb'))
print("Model training complete and saved!")
