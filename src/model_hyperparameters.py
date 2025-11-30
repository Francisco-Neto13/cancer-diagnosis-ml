import os
import pandas as pd

def save_model_hyperparameters(models: dict, output_dir="results/hyperparameters"):
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for model_name, model_obj in models.items():
        params = model_obj.get_params()
        rows.append({
            "model": model_name,
            "hyperparameters": params
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/hyperparameters.csv", index=False)


# Exemplo (substituir pelos modelos treinados de verdade)
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    save_model_hyperparameters(models)
