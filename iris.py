import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset Preview:")
print(df.head())

# Plot data
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species_name', data=df, palette='Set1')
plt.title('Iris Flower Sepal Dimensions')
plt.show()

# Split and Train
X = df[iris_data.feature_names]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%") 
