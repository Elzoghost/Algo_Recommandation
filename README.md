Description :
Ce code implémente un système de recommandation basé sur la méthode de filtrage collaboratif. Le système utilise les évaluations de produits données par les utilisateurs pour trouver les produits les plus similaires aux produits préférés d'un utilisateur. Les produits les plus similaires sont ensuite recommandés à l'utilisateur.

Le code utilise Pandas pour lire les données d'évaluation des produits à partir de fichiers tsv. Les données sont nettoyées et converties en matrices de notes utilisateur-produit. La similarité entre les notes de tous les utilisateurs est calculée en utilisant la similarité cosinus.

Le code offre une méthode "recommend_products" pour recommander des produits à un utilisateur donné en fonction du nombre de recommandations souhaité.

Fonctionnement :
Le code commence par importer les bibliothèques Pandas, NumPy et sklearn. La classe CollaborativeFiltering est ensuite définie. Les objets de cette classe prennent en entrée deux fichiers tsv : un fichier de notes d'évaluation des utilisateurs pour les produits et un fichier de descriptions de produits.

La méthode  **init** () initialise les attributs de l'objet, y compris les noms de fichiers et le nombre de recommandations par défaut.

La méthode calculate_similarity_matrix() calcule la matrice de similarité en utilisant la similarité cosinus.

La méthode recommend_products() prend en entrée un ID utilisateur et un nombre de recommandations. Si le nombre de recommandations n'est pas spécifié, le nombre par défaut est utilisé. La méthode récupère les notes de l'utilisateur et calcule les produits les plus similaires. Les produits recommandés sont ensuite sélectionnés et renvoyés.

La méthode get_product_names() est utilisée pour obtenir les noms des produits recommandés à partir de leurs identifiants.

Enfin, si le code est exécuté directement, il crée un objet CollaborativeFiltering et recommande des produits à un utilisateur spécifié.

Limitations :
Le code a plusieurs limites. Tout d'abord, il ne prend pas en compte les préférences des utilisateurs en dehors des notes données pour les produits. De plus, il n'utilise pas de modèles pour prédire les évaluations manquantes. Enfin, le code ne traite pas les évaluations biaisées ou les faux avis.
