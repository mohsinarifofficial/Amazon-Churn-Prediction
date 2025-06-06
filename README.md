# Amazon-Churn-Prediction
This project focuses on building an end-to-end AI solution to predict customer churn using a hybrid feature set derived from both textual reviews and numerical metadata. The pipeline involves big data preprocessing with Apache Spark, advanced text vectorization, class balancing, and model training using PyTorch.

Key Features:

Big Data Preprocessing with Apache Spark:
Utilized PySpark to efficiently handle and preprocess large-scale customer review datasets, ensuring scalable and distributed data handling.

Feature Engineering:

Applied TF-IDF vectorization to extract semantic features from customer review summaries.

Combined vectorized text embeddings with structured metadata (e.g., HelpfulnessNumerator, Score) for a multi-modal input feature space.

Class Balancing with SMOTE:
Addressed the class imbalance in churn labels using Synthetic Minority Over-sampling Technique (SMOTE) to improve model generalization and performance.

Model Training:
Employed PyTorch to build and train a deep learning classification model, optimizing it to differentiate between churn and non-churn customers based on combined features.

Data Workflow:

Raw reviews cleaned and vectorized using feature_engg.py.

SMOTE applied using apply_smote.py for balancing.

Final model training done in training_model.py.

Technologies Used:
Apache Spark, PySpark, Python, Pandas, Scikit-learn, Imbalanced-learn (SMOTE), TF-IDF, PyTorch, NumPy

Outcome:
A robust, scalable customer churn prediction pipeline suitable for handling multi-modal and imbalanced data, capable of real-world deployment for e-commerce or telecom customer analytics.

