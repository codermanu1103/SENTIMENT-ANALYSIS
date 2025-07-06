# SENTIMENT-ANALYSIS

COMPANY NAME - CODTECH IT SOLUTIONS

NAME - Manthan Gupta

INTERN ID - CT06DG1412

DOMAIN NAME - DATA ANALYSIS

DURATION - 6 WEEKS(June 14th 2025 to July 29th 2025)

MENTOR - NEELA SANTHOSH KUMAR

Description:

🔹 1. Importing Libraries
The notebook starts by importing key Python libraries:
numpy and matplotlib for numerical operations and plotting.
sklearn modules for:
Loading datasets (load_files)
Preprocessing (TfidfVectorizer)
Modeling (LogisticRegression)
Evaluation (accuracy_score, confusion_matrix, classification_report)

🔹 2. Loading the IMDb Dataset
Loads the IMDb movie reviews dataset from the aclImdb/train folder.
It uses only two categories: pos (positive) and neg (negative).
The text data (X) and labels (y) are extracted from files.
Prints the total number of positive and negative reviews loaded.

🔹 3. Text Vectorization using TF-IDF
Applies TfidfVectorizer to convert reviews into a numeric format suitable for ML models.
Uses a vocabulary of the top 5,000 features and removes English stopwords.
Splits the dataset into:
X_train, y_train for training
X_val, y_val for validation (80-20 split)

🔹 4. Training the Logistic Regression Model
Initializes a LogisticRegression model with max_iter=1000.
Fits the model using the training data.
Predicts sentiments (y_pred) on the validation set.

🔹 5. Evaluating the Model
Calculates and prints:
Accuracy Score of the predictions.
Classification Report (precision, recall, F1-score).
Confusion Matrix to visualize prediction performance.

🔹 6. Identifying Important Words
Extracts the model’s coefficients for each word (feature).
Identifies:
Top 10 positive words with the highest coefficients.
Top 10 negative words with the lowest coefficients.
Helps interpret which words strongly influence sentiment predictions.

🔹 7. Testing on Unseen IMDb Data
Loads aclImdb/test data (new, unseen reviews).
Transforms the test reviews using the same TF-IDF vectorizer.
Predicts sentiments on the test set.
Prints the test accuracy, showing how well the model generalizes.

🔹 8. Output & Insights
The model achieves solid performance on both validation and test sets.
Positive and negative influential words provide explainability.
The notebook shows a full ML pipeline for sentiment classification.

📚 Dataset Acknowledgement

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the ACL: Human Language Technologies, 142–150. ACL Anthology

Output:

<img width="775" height="472" alt="Image" src="https://github.com/user-attachments/assets/29809211-2bdd-4485-8ef8-9f1c71f50ca7" />

<img width="1302" height="192" alt="Image" src="https://github.com/user-attachments/assets/596753a6-11f1-4409-abc0-063fda4d03c2" />

<img width="448" height="55" alt="Image" src="https://github.com/user-attachments/assets/6ac9dae2-2a9a-4aee-bd29-dbcdb715fe9b" />
