import pandas as pd
from sklearn.datasets import load_breast_cancer
import joblib
import os


def load_raw_data(file_path='data/breast_cancer.csv'): # Função principal para carregar o dataset de câncer de mama.
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset carregado de: {file_path}")
        return df

    except FileNotFoundError:
        print(f"Arquivo não encontrado em {file_path}. Carregando dataset nativo do scikit-learn...")
        
        cancer = load_breast_cancer(as_frame=True)
        df = cancer.frame
        
        # Renomeia a coluna 'target' para 'diagnosis' para compatibilidade com o restante do projeto
        if 'target' in df.columns:
            df.rename(columns={'target': 'diagnosis'}, inplace=True)
        
        print("Dataset carregado com sucesso do scikit-learn.")
        return df

def load_test_data(data_dir='models'): # Carrega os dados de teste (X_test e y_test) salvos após o pré-processamento.
    file_path = os.path.join(data_dir, 'test_data.joblib')
    try:
        X_test, y_test = joblib.load(file_path)
        print(f"Dados de teste carregados de: {file_path}")
        return X_test, y_test
    except FileNotFoundError:
        print(f"ERRO: Dados de teste não encontrados em {file_path}. Execute o treinamento primeiro.")
        return None, None