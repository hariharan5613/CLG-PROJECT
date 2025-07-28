import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("MultiModal Data.csv")

drop_cols = ['sex', 'ethinicity', 'jaundice', 'autism', 'region',
             'used_app_before', 'age_desc', 'relation', 'ASD1', 'ASD2']
df = df.drop(columns=drop_cols)

eeg_cols = [col for col in df.columns if 'Power' in col or 'SampE' in col]
questionnaire_cols = [col for col in df.columns if col.startswith('A') or col.startswith('Q')]
feature_cols = eeg_cols + questionnaire_cols

df = df[feature_cols + ['Diagnosis']]
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}
grid = GridSearchCV(GradientBoostingClassifier(), params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (EEG + Questionnaire)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# User input
print("\nEnter EEG and questionnaire responses:")
user_input = []
for col in X.columns:
    val = float(input(f"{col}: "))
    user_input.append(val)

result = best_model.predict([user_input])[0]
prob = best_model.predict_proba([user_input])[0][1] * 100

if result == 1:
    print(f"\n Likely signs of Autism. Confidence: {prob:.2f}%")
else:
    print(f"\n Unlikely signs of Autism. Confidence: {prob:.2f}%")
