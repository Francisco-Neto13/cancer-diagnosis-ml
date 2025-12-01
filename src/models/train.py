import joblib
import os 
from data.loader import load_raw_data
from data.preprocess import preprocess_data
from .model_hyperparameters import get_base_models


def train_models(X_train, y_train, models): # Função que treina cada um dos modelos no dicionário 'models'.
    trained_models = {}
    print("\n## INICIANDO O TREINAMENTO DOS MODELOS ##")
    for name, model in models.items():
        print(f"Treinando {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"{name} treinado com sucesso.")
        except Exception as e:
            print(f"ERRO: Falha ao treinar {name}. Detalhe: {e}")
            
    return trained_models

def save_models_and_artifacts(trained_models, X_test, y_test, scaler, output_dir='models'):
    """Função que salva os modelos treinados, o scaler e os dados de teste."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n## SALVANDO ARTEFATOS (MODELOS, SCALER, TEST DATA) ##")
    
    # Salvar Modelos
    for name, model in trained_models.items():
        file_path = os.path.join(output_dir, f'{name}_model.joblib')
        joblib.dump(model, file_path)
        print(f"Modelo {name} salvo em: {file_path}")
    
    # Salvar o scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print("Scaler salvo em: models/scaler.joblib")
    
    # Salvar Dados de Teste
    joblib.dump((X_test, y_test), os.path.join(output_dir, 'test_data.joblib'))
    print("Dados de teste salvos em: models/test_data.joblib")


def run_training(): # Executa a pipeline completa de carregamento, pré-processamento, treinamento e salvamento.
    
    # Carregar e Pré-processar Dados
    df = load_raw_data() # Nova função do src/data/loader.py
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df) 
    
    # Obter e Treinar Modelos
    models_to_train = get_base_models() 
    trained_models = train_models(X_train, y_train, models_to_train)
    
    # Salvar Artefatos
    save_models_and_artifacts(trained_models, X_test, y_test, scaler)
    
    return trained_models

if __name__ == '__main__':
    run_training()
    print("\nTreinamento e salvamento concluídos.")