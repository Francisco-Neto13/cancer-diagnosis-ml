import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess_data
import os 


def get_models(): # Função que retorna um dicionário com os modelos que serão treinados
    models = {
        # Regressão Logística
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000), 
        # K-Nearest Neighbors
        'KNN': KNeighborsClassifier(n_neighbors=5), 
        # Support Vector Machine (Kernel RBF)
        'SVC_RBF': SVC(kernel='rbf', C=10, gamma='auto', random_state=42, probability=True), 
        # Random Forest
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    return models

def train_models(X_train, y_train, models): # Função que treina cada um dos modelos
    trained_models = {}
    print("\n## INICIANDO O TREINAMENTO DOS MODELOS ##")
    for name, model in models.items():
        print(f"Treinando {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} treinado com sucesso.")
    
    return trained_models

def save_models(trained_models, output_dir='models'): # Função que salva cada um dos modelos treinados
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n## SALVANDO MODELOS ##")
    for name, model in trained_models.items():
        file_path = os.path.join(output_dir, f'{name}_model.joblib')
        joblib.dump(model, file_path)
        print(f"Modelo {name} salvo em: {file_path}")

if __name__ == '__main__':
    # 1. Carregar e Pré-processar Dados
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # 2. Obter e Treinar Modelos
    models_to_train = get_models()
    trained_models = train_models(X_train, y_train, models_to_train)
    
    # 3. Salvar Modelos, Scaler e Dados de Teste
    save_models(trained_models)
    
    # Salvar o scaler (necessário para pré-processar novos dados)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Scaler salvo em: models/scaler.joblib")
    
    # Salva os dados de teste para a avaliação
    joblib.dump((X_test, y_test), 'models/test_data.joblib')
    print("Dados de teste salvos em: models/test_data.joblib")