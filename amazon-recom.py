import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate


class CollaborativeFiltering:
    def __init__(self, ratings_file, products_file, default_num_recommendations=10):
        self.ratings_file = ratings_file
        self.products_file = products_file
        self.ratings = pd.read_csv(ratings_file, sep='\t', header=0, names=['userId', 'productId', 'rating', 'timestamp'], error_bad_lines=False)
        self.ratings.drop('timestamp', axis=1, inplace=True)
        self.products = pd.read_csv(products_file, sep='\t', header=0, names=['productId', 'title', 'description'], error_bad_lines=False)
        self.similarity_matrix = None
        self.default_num_recommendations = default_num_recommendations

    def preprocess_ratings(self):
        # Créer une matrice de notes utilisateur-produit
        self.ratings_matrix = self.ratings.pivot_table(index='userId', columns='productId', values='rating').fillna(0)
        # Standardiser les notes utilisateur (nous ne voulons pas que les utilisateurs qui donnent des notes élevées aient plus de poids)
        scaler = StandardScaler()
        self.ratings_matrix = scaler.fit_transform(self.ratings_matrix)
        # Remplir les notes manquantes en utilisant un SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        self.ratings_matrix = imputer.fit_transform(self.ratings_matrix)
        # Réduire la dimension de la matrice à l'aide d'une décomposition en valeurs singulières tronquées (TruncatedSVD)
        svd = TruncatedSVD(n_components=50)
        self.ratings_matrix = svd.fit_transform(self.ratings_matrix)

    def calculate_similarity_matrix(self):
        self.similarity_matrix = cosine_similarity(self.ratings_matrix, self.ratings_matrix)

    def recommend_products(self, user_id, num_recommendations=None):
        if num_recommendations is None:
            num_recommendations = self.default_num_recommendations
        if self.similarity_matrix is None:
            self.preprocess_ratings()
            self.calculate_similarity_matrix()

        user_ratings = self.ratings_matrix[user_id]
        # Prédire les notes manquantes à l'aide d'un modèle de régression
        missing_ratings_indices = np.where(user_ratings == 0)[0]
        if len(missing_ratings_indices) > 0:
            X = np.delete(self.ratings_matrix, user_id, axis=0)
            y = self.ratings_matrix[:, missing_ratings_indices].flatten()
            y = np.delete(y, user_id, axis=0)
            model = Ridge(alpha=1)
            model.fit(X, y)
            predicted_ratings = model.predict(self.ratings_matrix[user_id].reshape(1, -1))[0]
            user_ratings[missing_ratings_indices] = predicted_ratings[missing_ratings_indices]
        # Corriger les évaluations biaisées ou les faux avis à l'aide de l'algorithme de 
        # factorisation de matrice SVD implémenté dans la bibliothèque surprise
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(dataframe=self.ratings[['userId', 'productId', 'rating']], reader=reader)
        svd = SVD()
        trainset = data.build_full_trainset()
        svd.fit(trainset)
        user_ratings = [svd.predict(user_id, product_id)[3] for product_id in self.products['productId']]
        recommended_indices = np.argsort(user_ratings)[::-1][:num_recommendations]
        recommendations = self.products.iloc[recommended_indices]
        return recommendations

# Exemple d'utilisation de la classe CollaborativeFiltering

if __name__ == '__main__':
    cf = CollaborativeFiltering(ratings_file='ratings.dat', products_file='products.dat', default_num_recommendations=10)
    recommendations = cf.recommend_products(user_id=1, num_recommendations=5)
    print(recommendations)







"""import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    def __init__(self, ratings_file):
        self.ratings = pd.read_csv(ratings_file, sep='\t', header=0, names=['userId', 'productId', 'rating', 'timestamp'], error_bad_lines=False)
        self.ratings.drop('timestamp', axis=1, inplace=True)
        self.similarity_matrix = None

    def calculate_similarity_matrix(self):
        self.ratings_matrix = self.ratings.pivot_table(index='userId', columns='productId', values='rating').fillna(0)
        self.similarity_matrix = cosine_similarity(self.ratings_matrix, self.ratings_matrix)

    def recommend_products(self, user_id, num_recommendations=10):
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        user_ratings = self.ratings_matrix.loc[user_id].values.reshape(1, -1)
        similarities = self.similarity_matrix[user_ratings.shape[0]-1]
        similar_users_indices = similarities.argsort()[:-num_recommendations-1:-1]
        similar_users_ratings = self.ratings_matrix.iloc[similar_users_indices]
        product_ratings = similar_users_ratings.stack().reset_index(name='rating')
        product_ratings = product_ratings[product_ratings['rating'] > 0]
        product_ratings = product_ratings[~product_ratings['productId'].isin(user_ratings)]
        top_products = product_ratings.groupby('productId')['rating'].mean().sort_values(ascending=False).head(num_recommendations).index.tolist()

        return top_products

if __name__ == '__main__':
    cf = CollaborativeFiltering('amazon_reviews_us_Electronics_v1_00.tsv')
    recommendations = cf.recommend_products ('userId')
#(1935753)
"""
