from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Load and clean the data
df = pd.read_csv("Cleaned_and_ImabalacedReviews.csv")
df = df.drop(df.columns[0], axis=1)  # Drop the unnamed index column if present

# Step 1: Vectorize text (reduced features and fill NaNs)
tfidf = TfidfVectorizer(max_features=300)  # Reduced to save memory
summary_embeddings = tfidf.fit_transform(df['Summary'].fillna("")).toarray().astype(np.float32)

# Optional: Save embeddings to inspect (heavy, avoid if not needed)
df['Summary_embeddings'] = summary_embeddings.tolist()

print(df.info())
df.to_csv("ewdwedwede.csv",index=False)


# # Step 2: Prepare numeric features
numeric_features = df[['ProductId', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score']].values.astype(np.float32)

# Combine embeddings and numeric features
X = np.hstack((summary_embeddings, numeric_features))

# Target variable
y = df['Churn label'].values

# Step 3: Apply SMOTE (on smaller, float32 data)
sm = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Optional: Save resampled data (e.g., for training or inspection)
resampled_df = pd.DataFrame(X_resampled)
resampled_df['Churn label'] = y_resampled
resampled_df.to_csv("resampled_data.csv", index=False)
