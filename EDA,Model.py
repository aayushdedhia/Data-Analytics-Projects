"""Online Shoppers Purchasing Intention Project Pipeline

This script loads the dataset, preprocesses features, trains a Gradient Boosting classifier,
evaluates performance, and saves predictions and the processed dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Load data
df = pd.read_csv('online_shoppers_intention.csv')

# Feature lists
numeric_features = {numeric_features}
categorical_features = {categorical_features}

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
])

# Split
X = df[numeric_features + categorical_features]
y = df['Revenue'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)

# Evaluation
y_test_proba = pipeline.predict_proba(X_test)[:, 1]
print(f"Test ROC AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print(classification_report(y_test, pipeline.predict(X_test)))

# Full dataset predictions
df['Predicted_Probability'] = pipeline.predict_proba(df[numeric_features + categorical_features])[:,1]
df['Predicted_Label'] = pipeline.predict(df[numeric_features + categorical_features])

# Save processed dataset
df.to_csv('online_shoppers_processed.csv', index=False)
