import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration

@tf.keras.utils.register_keras_serializable()
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.place_embedding = layers.Embedding(
            num_places,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.place_bias = layers.Embedding(num_places, 1)

        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_vector = self.dropout(user_vector)

        user_bias = self.user_bias(inputs[:, 0])
        place_vector = self.place_embedding(inputs[:, 1])
        place_vector = self.dropout(place_vector)

        place_bias = self.place_bias(inputs[:, 1])

        dot_user_place = tf.reduce_sum(user_vector * place_vector, axis=1, keepdims=True)
        x = dot_user_place + user_bias + place_bias
        return tf.squeeze(x, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'num_places': self.num_places,
            'embedding_size': self.embedding_size,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
destinasi_df = pd.read_csv('data/destinasi_df.csv')
rating_df = pd.read_csv('data/rating_df.csv')
cb_df = pd.read_csv('data/cb_df.csv')

cosine_sim_df = joblib.load('models/cosine_sim_df.pkl')
model_cf = tf.keras.models.load_model(
    'models/collab_model.keras',
    custom_objects={'RecommenderNet': RecommenderNet}
)
user_to_user_encoded = joblib.load('models/user_to_user_encoded.pkl')
place_to_place_encoded = joblib.load('models/place_to_place_encoded.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
tfidf_matrix = tfidf_vectorizer.transform(cb_df['Combined_Features'])

def content_based_recommendations(place_name, similarity_data=cosine_sim_df, items=cb_df, k=5):

    if place_name not in items['Place_Name'].values:
        return pd.DataFrame()

    index = items[items['Place_Name'] == place_name].index[0]
    sim_scores = list(enumerate(similarity_data.iloc[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]
    place_indices = [i[0] for i in sim_scores]
    place_ids = items.iloc[place_indices]['Place_Id'].tolist()

    return place_ids

def collaborative_filtering_recommendations(user_id, n=5):

    if user_id not in user_to_user_encoded:
        return pd.DataFrame()

    user_encoded = user_to_user_encoded[user_id]
    place_ids = rating_df['Place_Id'].unique()
    visited_places = rating_df[rating_df['User_Id'] == user_id]['Place_Id']
    place_ids_unvisited = [p for p in place_ids if p not in visited_places]
    place_encoded_unvisited = [
        place_to_place_encoded[p] for p in place_ids_unvisited
        if p in place_to_place_encoded
    ]

    user_place_array = np.array([[user_encoded, p_enc] for p_enc in place_encoded_unvisited])
    ratings = model_cf.predict(user_place_array).flatten()
    top_ratings_indices = ratings.argsort()[-n:][::-1]
    recommended_place_ids = [place_ids_unvisited[i] for i in top_ratings_indices]

    return recommended_place_ids

def get_travel_recommendations(user_id, favorite_place=None):

    all_recommendations = []
    cf_recs = collaborative_filtering_recommendations(user_id)
    all_recommendations.extend(cf_recs)

    if favorite_place:
        cb_recs = content_based_recommendations(favorite_place)
        all_recommendations.extend(cb_recs)

    unique_recommendations = list(set(all_recommendations))
    recommendations_df = destinasi_df[
        destinasi_df['Place_Id'].isin(unique_recommendations)
    ].copy()

    recommendations_df['Recommendation_Source'] = 'Hybrid'
    recommendations_df.loc[
        recommendations_df['Place_Id'].isin(cf_recs), 'Recommendation_Source'
    ] = 'Collaborative'

    if favorite_place:
        recommendations_df.loc[
            recommendations_df['Place_Id'].isin(cb_recs), 'Recommendation_Source'
        ] = 'Content-Based'

    return recommendations_df


#IMPLEMENTASI DENGAN MENGGGABUNGKAN 2 PENDEKATAN YANG LEBIH FLEKSIBEL

new_user_recs = get_travel_recommendations(user_id=1)
user_recs = get_travel_recommendations(
    user_id= 3,
    favorite_place= "Monumen Nasional"
)

print("Rekomendasi untuk user dengan favorite place 'Monumen Nasional':")
from IPython.display import display
display(user_recs)

print("Rekomendasi untuk user baru (tanpa favorite place):")
display(new_user_recs)



#IMPLEMENTASI HANYA BERDASARKAN CONTENT DESTINASINYA DENGAN INPUT KATEGORI NAMA ATAU KOTA

def infer_cbf_search(query, top_k=10):
    """
    Fungsi inference Content-Based Filtering menggunakan cosine similarity
    antara query dan TF-IDF matrix dari Combined_Features.
    Juga menyesuaikan skor berdasarkan City & Category.
    """
    weight_city = 0.15
    weight_category = 0.05

    query = query.lower().strip()
    keywords = query.split()

    query_vec = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_k * 3]

    unique_cities = cb_df['City'].str.lower().unique().tolist()
    city_in_query = [c for c in unique_cities if c in query]

    recommendations = []
    for idx in top_indices:
        place = cb_df.iloc[idx]
        base_score = similarity_scores[idx]
        adjusted_score = base_score

        if city_in_query and place['City'].lower() in city_in_query:
            adjusted_score += weight_city

        if any(kw in place['Category'].lower() for kw in keywords):
            adjusted_score += weight_category

        rec = place[['Place_Id']].copy()
        rec['Similarity_Score'] = round(adjusted_score, 4)
        rec['Search_Match'] = query
        recommendations.append(rec)

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values('Similarity_Score', ascending=False)
    rec_df = rec_df.drop_duplicates(subset=['Place_Id']).head(top_k)

    merged_df = pd.merge(rec_df, destinasi_df, on='Place_Id', how='left')
    return merged_df.to_dict(orient='records')

hasil = infer_cbf_search(" wisata alam daerah yogyakarta ", top_k=5)
display(hasil)
for item in hasil:
    print(f"\nNama: {item['Place_Name']}")
    print(f"Kategori: {item['Category']}")
    print(f"Kota: {item['City']}")
    print(f"Skor Similarity: {item['Similarity_Score']:.4f}")
    print(f"Match dengan: {item['Search_Match']}")
    print(f"Deskripsi: {item['Description'][:100]}...")

#GENERATIVE AI UNTUK TEKS REKOMENDASI SINGKAT

model_dir = "t5-finetuned-recommendation-final"
tokenizer = T5Tokenizer.from_pretrained(model_dir, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

rekom_texts = []
for _, row in user_recs.iterrows():
    teks = f"{row['Place_Name']} di {row['City']}, kategori {row['Category']}, rating {row['Rating']}"
    rekom_texts.append(teks)
input_text = "Rekomendasi tempat wisata: " + "; ".join(rekom_texts)

def generate_natural_recommendation(user_id, favorite_place=None, top_n=1):

    user_recs = get_travel_recommendations(user_id=user_id, favorite_place=favorite_place)

    if user_recs.empty:
        return "Tidak ada rekomendasi tersedia untuk user ini."

    user_recs = user_recs.head(top_n)
    input_template = "User menyukai kategori: {category}; lokasi: {city}; tempat: {place}; rating: {rating}"

    parts = []
    for _, row in user_recs.iterrows():
        part = input_template.format(
            category=row['Category'],
            city=row['City'],
            place=row['Place_Name'],
            rating=row['Rating']
        )
        parts.append(part)

    input_text = " ; ".join(parts)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result_text

hasil = generate_natural_recommendation(user_id=1,favorite_place="Monumen Nasional")
print(hasil)