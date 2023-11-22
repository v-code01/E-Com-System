import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import cross_validate

import scikit-learn
import fastapi
import docker

# Load the MovieLens dataset (or any other dataset you have)
# Here, I'm using a small version of the MovieLens dataset
data = Dataset.load_builtin('ml-100k')

# Create a surprise reader to parse the dataset
reader = Reader(line_format='user item rating timestamp', sep='\t')

# Load the data using the reader
data = Dataset.load_from_file('path/to/u.data', reader=reader)

# Build the collaborative filtering model (user-based)
sim_options = {
    'name': 'cosine',
    'user_based': True
}

model = KNNBasic(sim_options=sim_options)

# Evaluate the model using cross-validation
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the model on the entire dataset
trainset = data.build_full_trainset()
model.fit(trainset)

# Generate recommendations for a specific user (user id = 1)
user_id = '1'
user_items = trainset.ur[trainset.to_inner_uid(user_id)]
user_unrated_items = [item for item in trainset.all_items() if item not in user_items]

# Predict ratings for unrated items
predictions = [model.predict(user_id, item) for item in user_unrated_items]

# Get the top N recommendations
N = 5
top_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:N]

# Print the top recommendations
for recommendation in top_recommendations:
    print(f"Item ID: {trainset.to_raw_iid(recommendation.iid)} | Estimated Rating: {recommendation.est}")
