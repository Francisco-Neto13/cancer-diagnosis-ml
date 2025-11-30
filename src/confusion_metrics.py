import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def save_confusion_matrix(y_true, y_pred, model_name, output_dir="results/confusion_matrices"):
    os.makedirs(output_dir, exist_ok=True)

    # Criar matriz
    cm = confusion_matrix(y_true, y_pred)

    # Salvar como CSV
    df_cm = pd.DataFrame(cm, index=["Benigno", "Maligno"], columns=["Pred Benigno", "Pred Maligno"])
    df_cm.to_csv(f"{output_dir}/{model_name}_cm.csv")

    # Plotar e salvar imagem
    plt.figure(figsize=(5,4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{output_dir}/{model_name}_cm.png", dpi=300)
    plt.close()


# Exemplo de uso (substituir pelos dados reais)
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    model = RandomForestClassifier().fit(X_train, y_train)
    preds = model.predict(X_test)

    save_confusion_matrix(y_test, preds, "RandomForest")
