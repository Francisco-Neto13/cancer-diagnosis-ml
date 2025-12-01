import joblib
import os

def load_trained_models(model_names, model_dir='models'): #  Carrega todos os modelos joblib do diretório.
    loaded_models = {}
    print("\n## CARREGANDO MODELOS TREINADOS ##")
    for name in model_names:
        file_path = os.path.join(model_dir, f'{name}_model.joblib')
        try:
            loaded_models[name] = joblib.load(file_path)
            print(f"Modelo {name} carregado com sucesso.")
        except FileNotFoundError:
            print(f"AVISO: Arquivo {file_path} nao encontrado. Pulando este modelo.")
    return loaded_models