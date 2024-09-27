import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')
import nltk
nltk.download('wordnet')
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd #
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('/content/top_100_books.csv', encoding='latin-1')
df1.dataframeName = 'nyt_bestsellers.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1.shape
df1.info()
# Changes the column names
df1.columns = ['SI_NO','Rank_no', 'Book_Names', 'Author_Names', 'Ratings', 'Reviews', 'Type','Price']
print(df1.head())
# Create the heatmap of missing values
sns.heatmap(df1.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values in the Dataset')
# Add labels to the x-axis with the name of the columns
plt.xticks(range(len(df1.columns)), df1.columns, rotation=45)

# Show the plot
plt.show()
# Check the most reviews
most_review_book = df1.sort_values(by='Reviews', ascending=False).iloc[0]

print("Book with the most Reviews:")
# print("Title:", most_stars_book['title'])
print("Reviews:", most_review_book['Reviews'])
# Check the books which have the most reviews
N = 10

# Sort the DataFrame based on the number of reviews in descending order
sorted_books_df = df1.sort_values(by='Reviews', ascending=False)

# Get the top N books with the most reviews
top_books = sorted_books_df.head(N)

# Print the top N books with the most reviews
print("Top", N, "books with the most reviews:")
for index, book in top_books.iterrows():
    print("Book_Names:", book['Book_Names'])
    print("Number of Reviews:", book['Reviews'])
    print()  # Empty line for readability
#Check the Most reviews in barplots
N = 10

# Sort the DataFrame based on the number of reviews in descending order
sorted_books_df = df1.sort_values(by='Reviews', ascending=False)

# Get the top N books with the most reviews
top_books = sorted_books_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(top_books['Book_Names'], top_books['Reviews'], color='skyblue')
plt.xlabel('Book Title')
plt.ylabel('Number of Reviews')
plt.title('Top {} Books with the Most Reviews'.format(N))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# Assuming 'Book_Names' contains the book names and 'Ratings' contains the ratings
first_five_books = df1[['Book_Names', 'Ratings']].head()

plt.figure(figsize=(8, 6))
plt.bar(first_five_books['Book_Names'], first_five_books['Ratings'])
plt.xlabel('Book Names')
plt.ylabel('Ratings')
plt.title('Ratings of First Five Books')
plt.xticks(rotation=45)
plt.show()



# Assuming 'price' is the target variable and other columns are features
# Replace 'price' with the actual name of your target variable
X = df1.drop(columns=['Price'])  # Features
y = df1['Price']  # Target variable

# Perform one-hot encoding for categorical variables
# Assuming 'author' is a categorical variable, replace it with the actual name of the categorical variable
X_encoded = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Choose a Model (Linear Regression in this example)
model = RandomForestRegressor()


# Train the Model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_squared = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r_squared)

#Review Analysis of books
df=pd.read_csv("/content/customer reviews.csv")
df.head()
df["review description"] = df["review description"].astype(str)

# Tokenization
df['tokens'] = df["review description"].apply(word_tokenize)
df['token_length'] = df['tokens'].apply(len)

# Word frequencies
all_tokens = [token for sublist in df['tokens'] for token in sublist]
freq_dist = FreqDist(all_tokens)

# Plot the top 20 tokens
freq_dist.plot(20)

# Display the plot
plt.show()
# Example reviews
dataset = df["review description"]
# Analyze sentiment for each review
sentiment_labels = ["positive","Negative","Neutral"]
for review in dataset:
  analysis = TextBlob(review)
  polarity = analysis.sentiment.polarity
# Classify the sentiment
  if polarity > 0:
    sentiment_labels.append('Positive')
  elif polarity < 0:
    sentiment_labels.append('Negative')
  else:
    sentiment_labels.append('Neutral')
# Print the results
for review, sentiment in zip(dataset, sentiment_labels):
  print(f"Review: {review}\nSentiment: {sentiment}\n")
sentiment_counts = Counter(sentiment_labels)
# Print the counts
for sentiment, count in sentiment_counts.items():
  print(f"{sentiment}: {count} reviews")
  tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df["review description"].astype(str)).toarray()
# Assuming you want to predict sentiment or rating, use 'review rating' as the target variable
y = df["review description"]
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# Model Building
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Model Evaluation
y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
