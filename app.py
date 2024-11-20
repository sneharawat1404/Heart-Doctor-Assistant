from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__) 

# Load the trained model and encoders
model = pickle.load(open("modell.pkl", "rb"))

with open("dietmodel.pkl", "rb") as f:
    diet_model_data = pickle.load(f)
    diet_model = diet_model_data['model']
    diet_vectorizer = diet_model_data['vectorizer']
    diet_df = diet_model_data['df']
    diet_feature_df = diet_model_data['feature_df']

# with open("medicinemodel.pkl", "rb") as f:
#     medicine_model = pickle.load(f)


# Define mappings (replace these with your actual mappings)
sex_mapping ={'0': 0,'1': 1}
cp_mapping = {
    '0': 0, '1': 1, '2': 2, '3': 3
}
fbs_mapping = {
    '1': 1, '0': 0
}
restecg_mapping = {
    '0': 0, '1': 1, '2': 2
}
exang_mapping = {
    '0': 0, '1': 1
}
slope_mapping = {
    '0': 0, '1': 1, '2': 2
}
ca_mapping = {
    '0': 0, '1': 1, '2': 2, '3': 3
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def itsmymantra():
    return render_template('form.html')

@app.route('/medicine')
def medicine():
    return render_template('medicine.html')

@app.route('/dietpage')
def diet():
    return render_template('dietpage.html')

@app.route('/submit', methods=['POST'])
def predict():
    # Extracting data from form
    features = [
        int(request.form['age']),
        sex_mapping[request.form['sex']],  # Assuming 'sex' is already encoded or handled separately
        cp_mapping[request.form['cp']],
        int(request.form['trestbps']),
        int(request.form['chol']),
        fbs_mapping[request.form['fbs']],
        restecg_mapping[request.form['restecg']],
        int(request.form['thalach']),
        exang_mapping[request.form['exang']],
        float(request.form['oldpeak']),
        slope_mapping[request.form['slope']],
        ca_mapping[request.form['ca']],
        int(request.form['thal'])
    ]
    print("Features:", features)

    # Convert to numpy array
    final_features = [np.array(features)]

    # Predict using the loaded model
    prediction = model.predict(final_features)
    output = prediction[0]

    # Render result page
    return render_template('Report_result.html', data=output)

@app.route('/diet', methods=['POST'])
def submit_diet():
    nutrient = request.form.get('nutrient')
    disease = request.form.get('disease')
    diet = request.form.get('diet')
    eating_preference = request.form.get('eating_preference')

    user_input = {
        'Nutrient': nutrient,
        'Diet': diet,
        'Disease': disease,
        'Veg_Non': eating_preference
    }
    
    input_vector = create_input_vector(diet_vectorizer, diet_feature_df, user_input)
    
    # Predict using the model directly
    input_df = pd.DataFrame(input_vector)
    predictions = diet_model.kneighbors(input_df)
    
    recommendations = k_neighbor(diet_df, diet_feature_df, predictions)
    
    return render_template('diet_result.html', recommendations=recommendations.to_dict(orient='records'))

# @app.route('/recommend', methods=['POST'])
# def recommend_drug():
#     Description = request.form['Description']
#     results = recommended(medicine_model, Description)
#     return render_template('medicine.html', results=results)

def create_input_vector(vectorizer, feature_df, user_input):
    input_vector = np.zeros((1, feature_df.shape[1]))
    
    for feature, value in user_input.items():
        relevant_columns = [col for col in vectorizer.get_feature_names_out() if value in col]
        for col in relevant_columns:
            input_vector[0, feature_df.columns.get_loc(col)] = 1
    
    return input_vector

def k_neighbor(df, feature_df, predictions):
    indices = predictions[1][0]
    df_results = pd.DataFrame(columns=df.columns)
    for idx in indices:
        df_results = df_results.append(df.iloc[idx])
    df_results = df_results[['Name', 'description']]
    df_results = df_results.drop_duplicates(subset=['Name'])
    df_results = df_results.reset_index(drop=True)
    return df_results

# def recommended(model, Description):
#     input_vector = model['vectorizer'].transform([Description]).toarray()
#     input_similarity = cosine_similarity(input_vector, model['vector'])
#     distances = sorted(list(enumerate(input_similarity[0])), reverse=True, key=lambda x: x[1])
#     recommendations = [model['df'].iloc[i[0]].Drug_Name for i in distances[:5]]
#     return recommendations

if __name__ == "__main__":
    app.run(debug=True)


