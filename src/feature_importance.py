import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_feature_importance(model, feature_names, model_name, output_dir="results/feature_importance"):
    os.makedirs(output_dir, exist_ok=True)

    # Random Forest e árvores
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # Modelos lineares
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    else:
        print(f"Modelo {model_name} não possui importâncias interpretáveis.")
        return

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Salvando tabela
    df.to_csv(f"{output_dir}/{model_name}_importance.csv", index=False)

    # Gráfico
    plt.figure(figsize=(8,6))
    plt.barh(df["feature"][:15], df["importance"][:15])
    plt.gca().invert_yaxis()
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_importance.png", dpi=300)
    plt.close()


# Exemplo (usar seu modelo real no projeto)
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

    model = RandomForestClassifier().fit(X_train, y_train)
    save_feature_importance(model, data.feature_names, "RandomForest")
