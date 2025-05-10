# 1. Upload the Dataset
from google.colab import files
uploaded = files.upload()

# 2. Load the Dataset
import pandas as pd
df = pd.read_csv('15_fake_news_detection.csv')  # Update filename if different
df.head()

# 3. Data Exploration
print(df.info())
print(df.describe())
print("Shape:", df.shape)
print("Columns:", df.columns)

# 4. Check for Missing Values and Duplicates
print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

# 5. Visualize a Few Features
import seaborn as sns
import matplotlib.pyplot as plt

# Example: assuming 'label' is the target column
sns.countplot(data=df, x='label')  # Update if target column is different
plt.title("Target Distribution")
plt.show()

# 6. Identify Target and Features
target = 'label'  # Update to match your dataset
X = df.drop(columns=[target])
y = df[target]

# 7. Convert Categorical Columns to Numerical
categorical_cols = X.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_cols)

# 8. One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# 9. Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 11. Model Building
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# 12. Evaluation
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 13. Make Predictions from New Input
# Example: using the first row of the dataset
sample_input = X.iloc[0].values.reshape(1, -1)
print("Prediction for first sample:", model.predict(sample_input))

# 14. Convert to DataFrame and Encode
def preprocess_input(new_data_dict):
    new_df = pd.DataFrame([new_data_dict])
    new_df_encoded = pd.get_dummies(new_df)
    new_df_encoded = new_df_encoded.reindex(columns=X.columns, fill_value=0)
    return scaler.transform(new_df_encoded)

# 15. Predict the Final Grade (or Output)
# Example: replace with actual feature inputs
# new_data_dict = {"feature1": value, "feature2": value, ...}
# processed = preprocess_input(new_data_dict)
# prediction = model.predict(processed)

# 16. Deployment - Building an Interactive App
!pip install gradio
import gradio as gr

# 17. Create a Prediction Function
def predict_fake_news(text):
    # This assumes your dataset includes a 'text' column for news content
    input_df = pd.DataFrame({'text': [text]})
    input_df_encoded = pd.get_dummies(input_df)
    input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df_encoded)
    prediction = model.predict(input_scaled)
    return "Fake News" if prediction[0] == 1 else "Real News"

# 18. Create the Gradio Interface
interface = gr.Interface(
    fn=predict_fake_news,
    inputs="text",
    outputs="text",
    title="🎓Student Performance Predictor (Fake News Classifier Example)"
)
interface.launch()

# 19. 🎓 Student Performance Predictor
# If your actual goal is student performance prediction, update:
# - Dataset
# - Feature extraction
# - Target column
