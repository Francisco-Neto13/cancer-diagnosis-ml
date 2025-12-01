import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from data.loader import load_test_data
from models.model_utils import load_trained_models

#  Constantes de Caminho e Configuração
MODEL_NAMES_FOR_IMPORTANCE = ['LogisticRegression', 'RandomForest'] 
OUTPUT_DIR = "results/feature_importance"


def save_feature_importance(model, feature_names, model_name, output_dir=OUTPUT_DIR): # Calcula, salva a tabela e plota a importância das features para modelos compatíveis 
    os.makedirs(output_dir, exist_ok=True)

    # Obter as Importâncias
    if hasattr(model, "feature_importances_"):
        # Modelos baseados em árvores (e.g., Random Forest) usam importância baseada em Gini.
        importances = model.feature_importances_
        importance_type = "Baseado em Gini"
        
    elif hasattr(model, "coef_"):
        # Modelos lineares (e.g., Regressão Logística) usam o valor absoluto dos coeficientes.
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        importance_type = "Valor Absoluto do Coeficiente"
        
    else:
        print(f"AVISO: Modelo {model_name} não possui importâncias interpretáveis.")
        return

    # Criar e ordenar o DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    df['importance_type'] = importance_type

    # Salvando tabela 
    file_path_csv = os.path.join(output_dir, f'{model_name}_importance.csv')
    df.to_csv(file_path_csv, index=False, float_format="%.4f")
    print(f"Importância de Features (CSV) para {model_name} salva em: {file_path_csv}")

    # Gerar Gráfico
    N_TOP = 15
    top_features = df.head(N_TOP)
    
    plt.figure(figsize=(9, max(6, N_TOP * 0.4)))
    plt.barh(top_features["feature"], top_features["importance"], color='tab:green')
    plt.gca().invert_yaxis()
    plt.title(f"Importância de Features - {model_name} ({importance_type})")
    plt.xlabel("Importância")
    plt.tight_layout()
    
    file_path_png = os.path.join(output_dir, f'{model_name}_importance.png')
    plt.savefig(file_path_png, dpi=300)
    plt.close()
    print(f"Importância de Features (PNG) para {model_name} salva em: {file_path_png}")


def run_feature_importance_analysis():
    print("\n## INICIANDO ANÁLISE DE IMPORTÂNCIA DE FEATURES ##")
    
    # Carregar dados de teste
    X_test, _ = load_test_data()
    
    if X_test is None:
        return
    
    # Tenta obter os nomes das features do DataFrame X_test ou do scaler salvo.
    if not isinstance(X_test, pd.DataFrame):
        try:
            scaler = joblib.load('models/scaler.joblib')
            feature_names = list(scaler.feature_names_in_)
        except (FileNotFoundError, AttributeError):
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    else:
        feature_names = X_test.columns.tolist()

    # Carregar modelos relevantes do disco.
    models = load_trained_models(MODEL_NAMES_FOR_IMPORTANCE)

    if not models:
        print("Nenhum modelo compatível carregado para análise de importância.")
        return

    # Executar e Salvar Importâncias para cada modelo.
    for name, model in models.items():
        save_feature_importance(model, feature_names, name)

    print("## ANÁLISE DE IMPORTÂNCIA DE FEATURES CONCLUÍDA ##")