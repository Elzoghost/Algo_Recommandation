import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, ratings_file, products_file, default_num_recommendations=10):
        self.ratings_file = ratings_file
        self.products_file = products_file
        self.ratings = pd.read_csv(ratings_file, sep='\t', header=0, names=['customer_id', 'product_id', 'rating', 'timestamp'], on_bad_lines='skip')
        self.ratings.drop('timestamp', axis=1, inplace=True)
        self.products = pd.read_csv(products_file, sep='\t', header=0, names=['product_id', 'title', 'description'], error_bad_lines=False)
        self.similarity_matrix = None
        self.default_num_recommendations = default_num_recommendations

    def calculate_similarity_matrix(self):
        self.ratings_matrix = self.ratings.pivot_table(index='customer_id', columns='product_id', values='rating').fillna(0)
        self.similarity_matrix = cosine_similarity(self.ratings_matrix, self.ratings_matrix)

    def recommend_products(self, user_id, num_recommendations=None):
        if num_recommendations is None:
            num_recommendations = self.default_num_recommendations
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        user_ratings = self.ratings_matrix.loc[user_id].values.reshape(1, -1)
        similarities = self.similarity_matrix[user_ratings.shape[0]-1]
        similar_users_indices = similarities.argsort()[:-num_recommendations-1:-1]
        similar_users_ratings = self.ratings_matrix.iloc[similar_users_indices]
        product_ratings = similar_users_ratings.stack().reset_index(name='rating')
        product_ratings = product_ratings[product_ratings['rating'] > 0]
        product_ratings = product_ratings[~product_ratings['product_id'].isin(user_ratings)]
        top_products = product_ratings.groupby('product_id')['rating'].mean().sort_values(ascending=False).head(num_recommendations).index.tolist()

        top_products_names = self.get_product_names(top_products)

        return top_products_names

    def get_product_names(self, product_ids):
        product_names = self.products[self.products['product_id'].isin(product_ids)]['title'].tolist()
        return product_names

if __name__ == '__main__':
    cf = CollaborativeFiltering('amazon_reviews_us_Electronics_v1_00.tsv', 'amazon_reviews_us_Pet_Products_v1_00.tsv', default_num_recommendations=5)
    recommendations = cf.recommend_products(1935753, num_recommendations=5)
    print('Produits recommandés :', recommendations)