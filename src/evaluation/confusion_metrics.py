import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

RESULTS_DIR = "results/confusion_matrices" 


def save_confusion_matrix(y_true, y_pred, model_name, output_dir=RESULTS_DIR): # Calcula e salva a Matriz de Confusão como CSV e gera a visualização como PNG.
    
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    
    # Preparar DataFrame para visualização e CSV
    df_cm = pd.DataFrame(
        cm, 
        index=["Benigno (Real 0)", "Maligno (Real 1)"], 
        columns=["Pred Benigno (0)", "Pred Maligno (1)"]
    )

    # Salvar como CSV
    file_path_csv = os.path.join(output_dir, f'{model_name}_cm.csv')
    df_cm.to_csv(file_path_csv)
    print(f"Matriz de Confusão (CSV) salva em: {file_path_csv}")

    # Plotar e salvar imagem
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.tight_layout()
    
    file_path_png = os.path.join(output_dir, f'{model_name}_cm.png')
    plt.savefig(file_path_png, dpi=300)
    plt.close()
    print(f"Matriz de Confusão (PNG) salva em: {file_path_png}")