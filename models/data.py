import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('student-mat.csv', sep=';')
print(df.isnull().sum())

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

plt.figure(figsize=(8, 5))
plt.hist(df['G3'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Final Grade (G3)')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['studytime'], df['G3'], color='green')
plt.title('Study Time vs Final Grade')
plt.xlabel('Study Time')
plt.ylabel('Final Grade (G3)')
plt.grid(True)
plt.show()

X = df.drop('G3', axis=1)
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

joblib.dump(model, 'student_grade_predictor.joblib')
print("Model saved as student_grade_predictor.joblib")

joblib.dump(label_encoders, 'label_encoders.joblib')
print("Encoders saved as label_encoders.joblib")

joblib.dump(list(X.columns), 'feature_columns.joblib')
