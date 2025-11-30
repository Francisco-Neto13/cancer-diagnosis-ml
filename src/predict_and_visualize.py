import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from loader import load_all_models
from confusion_metrics import save_confusion_matrix

MODEL_NAMES = ['LogisticRegression', 'KNN', 'SVC_RBF', 'RandomForest']
RESULTS_DIR = "results"
PRED_CSV = os.path.join(RESULTS_DIR, "predictions_table.csv")
METRIC_PLOT = os.path.join(RESULTS_DIR, "model_accuracies.png")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Carregar dados de teste (gerados por train.py)
    try:
        X_test, y_test = joblib.load('models/test_data.joblib')
        print(f"Dados de teste carregados. X_test shape: {getattr(X_test, 'shape', 'unknown')}")
    except FileNotFoundError:
        print("ERRO: models/test_data.joblib nao encontrado. Execute `train.py` primeiro.")
        return

    # 2. Carregar modelos
    models = load_all_models(MODEL_NAMES)
    if not models:
        print("Nenhum modelo carregado. Verifique a pasta models/ e os nomes em MODEL_NAMES.")
        return

    # 3. Montar DataFrame de previsões
    preds_df = pd.DataFrame({'y_true': np.array(y_test).flatten()})

    accuracies = {}
    for name, model in models.items():
        # Previsão (labels)
        y_pred = model.predict(X_test)
        preds_df[f"{name}_pred"] = y_pred

        # Probabilidade (se disponível)
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

        # Salvar matriz de confusão usando função existente
        try:
            save_confusion_matrix(y_test, y_pred, name)
        except Exception as e:
            print(f"Falha ao salvar matriz de confusao para {name}: {e}")

        # Acurácia para o gráfico
        try:
            accuracies[name] = accuracy_score(y_test, y_pred)
        except Exception:
            accuracies[name] = np.nan

    # 4. Salvar tabela de previsões
    preds_df.to_csv(PRED_CSV, index=False)
    print(f"Tabela de previsões salva em: {PRED_CSV}")
    print(preds_df.head())

    # 5. Gráfico simples: acurácia por modelo
    try:
        names = list(accuracies.keys())
        vals = [accuracies[n] for n in names]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, vals, color='tab:blue')
        plt.ylim(0, 1)
        plt.ylabel("Acurácia")
        plt.title("Acurácia por Modelo (no conjunto de teste)")
        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(METRIC_PLOT, dpi=200)
        plt.close()
        print(f"Gráfico de acurácia salvo em: {METRIC_PLOT}")
    except Exception as e:
        print(f"Falha ao gerar/gravar gráfico: {e}")


if __name__ == "__main__":
    main()