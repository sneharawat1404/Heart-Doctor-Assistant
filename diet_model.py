import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

def get_features(df):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    nutrient_tfidf = vectorizer.fit_transform(df['Nutrient']).toarray()
    diet_tfidf = vectorizer.fit_transform(df['Diet']).toarray()
    disease_tfidf = vectorizer.fit_transform(df['Disease']).toarray()
    Veg_Non_tfidf = vectorizer.fit_transform(df['Veg_Non']).toarray()
    
    feature_df = pd.DataFrame(np.hstack([nutrient_tfidf, diet_tfidf, disease_tfidf, Veg_Non_tfidf]))
    return feature_df, vectorizer

def initialize_model(feature_df):
    model = NearestNeighbors(n_neighbors=40, algorithm='ball_tree')
    model.fit(feature_df)
    return model

def save_diet_model(df, filename='dietmodel.pkl'):
    feature_df, vectorizer = get_features(df)
    model = initialize_model(feature_df)
    data = {
        'model': model,
        'vectorizer': vectorizer,
        'df': df,
        'feature_df': feature_df
    }
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

if __name__ == '__main__':
    df_diet = pd.read_csv('Diet_dataset.csv')
    save_diet_model(df_diet)
    

