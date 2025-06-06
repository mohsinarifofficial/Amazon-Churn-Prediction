from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import ast

df = pd.read_csv("ewdwedwede.csv")
df["Summary_embeddings"] = df["Summary_embeddings"].apply(ast.literal_eval)

x = df.drop(["Summary", "Churn label"], axis=1)
summary_embeddings = np.array(x["Summary_embeddings"].tolist(), dtype=np.float32)
x = x.drop("Summary_embeddings", axis=1).reset_index(drop=True)
x = pd.concat([pd.DataFrame(summary_embeddings), x], axis=1)

y = df["Churn label"]
sm = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = sm.fit_resample(x, y)

X_resampled["Churn label"] = y_resampled
X_resampled.to_csv("resampled_data.csv", index=False)
