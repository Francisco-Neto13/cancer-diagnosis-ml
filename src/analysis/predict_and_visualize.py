import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from data.loader import load_test_data       
from models.model_utils import load_trained_models    

MODEL_NAMES = ['LogisticRegression', 'KNN', 'SVC_RBF', 'RandomForest']
RESULTS_DIR = "results"
PRED_CSV = os.path.join(RESULTS_DIR, "predictions_table.csv")
METRIC_PLOT = os.path.join(RESULTS_DIR, "model_accuracies.png")


def generate_predictions_table(models, X_test, y_test, output_path): # Gera a tabela de previsões (labels e probabilidades) e salva em CSV.
    preds_df = pd.DataFrame({'y_true': np.array(y_test).flatten()})
    accuracies = {}

    for name, model in models.items():
        # Previsão (labels)
        try:
            y_pred = model.predict(X_test)
            preds_df[f"{name}_pred"] = y_pred
            
            accuracies[name] = accuracy_score(y_test, y_pred)
            
        except Exception as e:
            print(f"AVISO: Falha ao gerar previsões de labels para {name}: {e}")
            accuracies[name] = np.nan
            preds_df[f"{name}_pred"] = np.nan
            continue

        # Probabilidade/Score 
        if hasattr(model, "predict_proba"):
            try:
                preds_df[f"{name}_prob"] = model.predict_proba(X_test)[:, 1]
            except Exception:
                preds_df[f"{name}_prob"] = np.nan
        elif hasattr(model, "decision_function"):
            try:
                preds_df[f"{name}_score"] = model.decision_function(X_test)
            except Exception:
                preds_df[f"{name}_score"] = np.nan
        
    # Salvar tabela
    preds_df.to_csv(output_path, index=False)
    print(f"Tabela de previsões salva em: {output_path}")
    print(preds_df.head())
    
    return accuracies


def plot_accuracies(accuracies, output_path): # Gera e salva o gráfico de barras de acurácia por modelo.
    try:
        names = list(accuracies.keys())
        vals = [accuracies[n] for n in names]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, vals, color='tab:blue')
        plt.ylim(0, 1)
        plt.ylabel("Acurácia")
        plt.title("Acurácia por Modelo (no conjunto de teste)")
        
        # Adicionar rótulos de valor em cima das barras
        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", 
                     ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Gráfico de acurácia salvo em: {output_path}")
    except Exception as e:
        print(f"Falha ao gerar/gravar gráfico de acurácia: {e}")


def run_prediction_analysis(): # Executa a pipeline de geração de previsões e plots iniciais de acurácia.
    print("\n## INICIANDO GERAÇÃO DE PREVISÕES E ANÁLISE INICIAL ##")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Carregar dados de teste
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    # Carregar modelos treinados
    models = load_trained_models(MODEL_NAMES)
    if not models:
        print("Nenhum modelo carregado. Análise de previsão abortada.")
        return

    # Gerar e salvar a tabela de previsões, retornando as acurácias
    accuracies = generate_predictions_table(models, X_test, y_test, PRED_CSV)

    # Plotar o gráfico de acurácia
    plot_accuracies(accuracies, METRIC_PLOT)
    
    print("## ANÁLISE DE PREVISÕES CONCLUÍDA ##")