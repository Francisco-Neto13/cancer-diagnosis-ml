from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_base_models() -> dict: # Retorna um dicionário com os modelos base instanciados e configurados.
    base_models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=5000), 
        
        'KNN': KNeighborsClassifier(n_neighbors=5), 
        
        'SVC_RBF': SVC(kernel='rbf', C=10, gamma='auto', random_state=42, probability=True), 
        
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    return base_models

def get_optimization_grid() -> dict: # Retorna um dicionário com a grade de hiperparâmetros para GridSearch ou RandomSearch.
    param_grid = {
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
    }
    return param_grid