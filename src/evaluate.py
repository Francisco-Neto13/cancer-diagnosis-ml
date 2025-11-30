import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from loader import load_all_models

def evaluate_models(models, X_test, y_test): ## Essa função calcula as métricas de avaliação de cada modelo. X e Y contêm valores binários 
    results = []
    print("\n## INICIANDO A AVALIACAO DOS MODELOS ##")

    # Garante que y_test é numérico (0 ou 1) para o cálculo das métricas
    y_test = np.array(y_test)

    # Definimos 1 como a classe positiva (Maligno)
    POS_LABEL = 1 

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Métricas de avaliação
        accuracy = accuracy_score(y_test, y_pred)
        # Usamos POS_LABEL=1 (Maligno) para as métricas de Precision, Recall e F1
        precision = precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Cálculo do ROC AUC
        roc_auc = 'N/A'
        try:
            # Tenta obter probabilidades (para RegLog, RF, KNN)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            # Tenta obter scores de decisão (para SVC)
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                raise AttributeError

            roc_auc = roc_auc_score(y_test, y_proba)

        except Exception:
            pass

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


def plot_roc_curves(models, X_test, y_test): # Essa função gera um gráfico de Curva para comparação. Assumindo também que X e Y contêm valores binários
    plt.figure(figsize=(10, 8))

    # Garante que y_test é numérico (0 ou 1)
    y_test = np.array(y_test)

    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

        except Exception:
            continue

    plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.50)')
    plt.xlabel('Taxa de Falso Positivo (False Positive Rate)')
    plt.ylabel('Taxa de Verdadeiro Positivo (True Positive Rate)')
    plt.title('Curvas ROC - Comparação de Modelos')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    model_names = ['LogisticRegression', 'KNN', 'SVC_RBF', 'RandomForest']

    # 1. Carregar Dados de Teste
    try:
        X_test, y_test = joblib.load('models/test_data.joblib')
        print(f"Dados de teste carregados. X_test shape: {X_test.shape}")
    except FileNotFoundError:
        print("ERRO: Dados de teste nao encontrados em models/test_data.joblib. Execute `train.py` primeiro.")
        exit()

    # 2. Carregar Modelos Treinados
    trained_models = load_all_models(model_names)

    if not trained_models:
        print("Nenhum modelo foi carregado para avaliacao.")
    else:
        # 3. Avaliar Modelos
        evaluation_df = evaluate_models(trained_models, X_test, y_test)

        # 4. Exibir Tabela de Resultados Comparativos
        print("\n## TABELA DE RESULTADOS COMPARATIVOS (METRICAS PRINCIPAIS) ##")
        print(
            evaluation_df[
                ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            ].sort_values(by='F1-Score', ascending=False).to_markdown(
                index=False, floatfmt=".4f"
            )
        )

        # 5. Plotar Curva ROC
        print("\nPlotando Curvas ROC...")
        plot_roc_curves(trained_models, X_test, y_test)