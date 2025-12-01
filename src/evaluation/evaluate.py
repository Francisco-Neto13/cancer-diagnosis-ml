import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

from data.loader import load_test_data          
from models.model_utils import load_trained_models     
from .confusion_metrics import save_confusion_matrix 

RESULTS_DIR = "results"


def evaluate_models(models, X_test, y_test, save_cm=True): # Calcula as métricas de avaliação de cada modelo e, opcionalmente, salva as Matrizes de Confusão.
    results = []
    print("\n## INICIANDO A AVALIACAO DOS MODELOS ##")

    y_test = np.array(y_test)
    POS_LABEL = 1 

    for name, model in models.items():

        y_pred = model.predict(X_test)

        # Cálculo das métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Cálculo do ROC AUC
        roc_auc = 'N/A'
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                raise AttributeError
            
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception as e:
            pass

        # Salvar Matriz de Confusão 
        if save_cm:
            print(f"\nGerando Matriz de Confusão para {name}...")
            save_confusion_matrix(y_test, y_pred, name)


        # Adicionar resultados
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Confusion Matrix': cm
        })

        print(f"\n--- Resultados para {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Matriz de Confusao:\n{cm}")

    return pd.DataFrame(results)


def plot_roc_curves(models, X_test, y_test, output_path=os.path.join(RESULTS_DIR, 'roc_curves_comparison.png')): # Gera um gráfico comparativo de Curva ROC e salva o PNG.
    print("\n## PLOTANDO CURVAS ROC ##")
    plt.figure(figsize=(10, 8))

    y_test = np.array(y_test)

    # Iterar e Plotar Curvas
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                continue

            # Calcula a curva e a área
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

        except Exception as e:
            print(f"AVISO: Não foi possível plotar Curva ROC para {name}. Erro: {e}")
            continue
            
    # Linha de base e finalização do plot
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.50)')
    plt.xlabel('Taxa de Falso Positivo (False Positive Rate)')
    plt.ylabel('Taxa de Verdadeiro Positivo (True Positive Rate)')
    plt.title('Curvas ROC - Comparação de Modelos')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Salvar
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico de Curvas ROC salvo em: {output_path}")


def save_evaluation_table(evaluation_df, output_path=os.path.join(RESULTS_DIR, 'comparative_table.csv')): # Salva a tabela de resultados comparativos em formato CSV e Markdown (TXT).
    
    # Selecionar e ordenar colunas principais (ordenando por F1-Score)
    df_clean = evaluation_df[
        ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    ].sort_values(by='F1-Score', ascending=False)
    
    # Salvar CSV
    df_clean.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\nTabela comparativa salva em CSV: {output_path}")
    
    # Salvar Markdown/TXT para fácil visualização do relatório
    markdown_table = df_clean.to_markdown(index=False, floatfmt=".4f")
    
    txt_path = os.path.join(RESULTS_DIR, 'comparative_table.txt')
    with open(txt_path, 'w') as f:
        f.write("## TABELA DE RESULTADOS COMPARATIVOS (MÉTRICAS PRINCIPAIS) ##\n\n")
        f.write(markdown_table)
    print(f"Tabela comparativa salva em TXT: {txt_path}")
    
    print("\n" + markdown_table)


def run_evaluation(model_names): # Orquestra o carregamento, avaliação e salvamento dos resultados.
    
    # Carregar Dados de Teste
    X_test, y_test = load_test_data()
    
    if X_test is None:
        return
    
    # Carregar Modelos Treinados
    trained_models = load_trained_models(model_names)

    if not trained_models:
        print("Nenhum modelo foi carregado para avaliacao. Avaliacao abortada.")
        return
    
    # Avaliar Modelos e obter o DataFrame de resultados
    evaluation_df = evaluate_models(trained_models, X_test, y_test)

    # Salvar Tabela de Resultados Comparativos (CSV e TXT)
    save_evaluation_table(evaluation_df)

    # Plotar Curva ROC
    plot_roc_curves(trained_models, X_test, y_test)
    
    # Retorna os modelos para uso posterior na pipeline (análise de erro, etc.)
    return trained_models


if __name__ == '__main__':
    model_names = ['LogisticRegression', 'KNN', 'SVC_RBF', 'RandomForest']
    
    run_evaluation(model_names)
    
    print("\nAvaliação de modelos concluída com sucesso.")