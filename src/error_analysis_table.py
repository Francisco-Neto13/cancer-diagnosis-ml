import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PRED_CSV = os.path.join(RESULTS_DIR, "predictions_table.csv")
ERRORS_CSV = os.path.join(RESULTS_DIR, "error_cases_by_model.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "error_summary_by_model.csv")
SUMMARY_PNG = os.path.join(RESULTS_DIR, "error_cases_summary.png")


def load_predictions(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    return pd.read_csv(path)


def compute_error_matrix(df):
    if "y_true" not in df.columns:
        raise KeyError("Coluna 'y_true' não encontrada em predictions_table.csv")

    error_rows = []
    summary = []

    for col in [c for c in df.columns if c.endswith("_pred")]:
        model = col[:-5]
        y_true = df["y_true"].values
        y_pred = df[col].values

        # normalize types for comparison
        try:
            y_true_cmp = y_true.astype(int)
            y_pred_cmp = y_pred.astype(int)
        except Exception:
            y_true_cmp = y_true
            y_pred_cmp = y_pred

        tp = int(((y_true_cmp == 1) & (y_pred_cmp == 1)).sum())
        tn = int(((y_true_cmp == 0) & (y_pred_cmp == 0)).sum())
        fp = int(((y_true_cmp == 0) & (y_pred_cmp == 1)).sum())
        fn = int(((y_true_cmp == 1) & (y_pred_cmp == 0)).sum())

        summary.append({"model": model, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "support": len(y_true)})
        # collect error rows for this model
        sel = df.loc[y_true_cmp != y_pred_cmp].copy()
        if not sel.empty:
            sel["model"] = model
            sel["y_pred"] = sel[col]
            sel["error_type"] = np.where((sel["y_pred"] == 1) & (sel["y_true"] == 0), "FP",
                                         np.where((sel["y_pred"] == 0) & (sel["y_true"] == 1), "FN", "other"))
            # keep sample index as id
            sel = sel.reset_index().rename(columns={"index": "sample_id"})
            error_rows.append(sel)

    summary_df = pd.DataFrame(summary).set_index("model")
    errors_df = pd.concat(error_rows, ignore_index=True) if error_rows else pd.DataFrame()
    return summary_df, errors_df


def plot_error_summary(summary_df, out_png):
    if summary_df.empty:
        print("Nenhum resumo de erros para plotar.")
        return

    df = summary_df.copy()
    models = df.index.tolist()
    fps = df["fp"].values
    fns = df["fn"].values

    x = np.arange(len(models))
    width = 0.6

    plt.figure(figsize=(max(6, len(models) * 1.2), 5))
    p1 = plt.bar(x, fps, width, label="FP", color="#d9534f")
    p2 = plt.bar(x, fns, width, bottom=fps, label="FN", color="#f0ad4e")

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Número de erros")
    plt.title("Erros por modelo (FP empilhado sobre FN)")
    plt.legend()

    # labels
    for i in range(len(models)):
        total = fps[i] + fns[i]
        if fps[i] > 0:
            plt.text(x[i], fps[i] / 2, str(fps[i]), ha="center", va="center", color="white", fontsize=9)
        if fns[i] > 0:
            plt.text(x[i], fps[i] + fns[i] / 2, str(fns[i]), ha="center", va="center", color="white", fontsize=9)
        plt.text(x[i], total + max(1, total * 0.03), str(total), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Gráfico de erros salvo em: {out_png}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        df = load_predictions(PRED_CSV)
    except FileNotFoundError as e:
        print(e)
        print("Execute predict_and_visualize.py primeiro para gerar predictions_table.csv")
        return

    summary_df, errors_df = compute_error_matrix(df)

    # salvar CSVs
    summary_df.to_csv(SUMMARY_CSV)
    if not errors_df.empty:
        errors_df.to_csv(ERRORS_CSV, index=False)
        print(f"Casos de erro salvos em: {ERRORS_CSV}")
    else:
        print("Nenhum caso de erro encontrado.")
    print(f"Resumo de erros salvo em: {SUMMARY_CSV}")

    # plot PNG
    plot_error_summary(summary_df, SUMMARY_PNG)


if __name__ == "__main__":
    main()