import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TESTDATA_PATH = os.path.join(MODELS_DIR, "test_data.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

OUT_FEATURES_CSV = os.path.join(RESULTS_DIR, "dataset_feature_stats.csv")
OUT_CLASS_CSV = os.path.join(RESULTS_DIR, "dataset_class_distribution.csv")
OUT_CORR_PNG = os.path.join(RESULTS_DIR, "correlation_matrix.png")
OUT_CLASS_PNG = os.path.join(RESULTS_DIR, "class_distribution.png")
OUT_HIST_PNG = os.path.join(RESULTS_DIR, "feature_histograms.png")


def load_test_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    data = joblib.load(path)
    # Expect tuple (X, y) or list-like [X, y]
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        return data[0], data[1]
    # If saved as dict
    if isinstance(data, dict) and "X" in data and "y" in data:
        return data["X"], data["y"]
    raise ValueError("Formato inesperado em test_data.joblib (esperado (X, y) ou {'X':..,'y':..})")


def get_feature_names(X):
    # Try to recover names from scaler if available
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            if hasattr(scaler, "feature_names_in_"):
                return list(scaler.feature_names_in_)
        except Exception:
            pass
    # If X is DataFrame, keep its columns
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    # Fallback
    n = X.shape[1] if hasattr(X, "shape") else (len(X[0]) if len(X) and hasattr(X[0], "__len__") else 0)
    return [f"feature_{i}" for i in range(n)]


def build_feature_stats(df):
    desc = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["median"] = df.median()
    desc["iqr"] = df.quantile(0.75) - df.quantile(0.25)
    desc["missing_count"] = df.isna().sum()
    desc["unique_count"] = df.nunique(dropna=False)
    desc["skew"] = df.skew()
    desc["kurtosis"] = df.kurtosis()
    # reorder useful columns
    cols = ["count", "missing_count", "unique_count", "mean", "std", "min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max", "median", "iqr", "skew", "kurtosis"]
    # ensure present
    cols_present = [c for c in cols if c in desc.columns]
    return desc[cols_present]


def plot_correlation(df, out_png):
    corr = df.corr()
    plt.figure(figsize=(max(6, len(corr) * 1), max(6, len(corr) * 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=False, cbar_kws={"shrink": .6})
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_class_distribution(y, out_png):
    ser = pd.Series(y).reset_index(drop=True)
    counts = ser.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    ax = counts.plot(kind="bar", color=sns.color_palette("pastel"))
    plt.ylabel("Count")
    plt.xlabel("Classe")
    plt.title("Distribuição por Classe")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_feature_histograms(df, out_png, ncols=4):
    n = df.shape[1]
    ncols = min(ncols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    figsize = (ncols * 3.0, max(3, nrows * 2.2))
    plt.figure(figsize=figsize)
    for i, col in enumerate(df.columns):
        ax = plt.subplot(nrows, ncols, i + 1)
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="tab:blue", ax=ax)
        ax.set_title(col)
        ax.tick_params(axis='x', labelrotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        X, y = load_test_data(TESTDATA_PATH)
    except Exception as e:
        print(e)
        return

    # convert X to DataFrame
    feature_names = get_feature_names(X)
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X, columns=feature_names)

    # feature stats
    feat_stats = build_feature_stats(df)
    feat_stats.to_csv(OUT_FEATURES_CSV)
    print(f"Estatísticas por feature salvas em: {OUT_FEATURES_CSV}")

    # class distribution
    y_ser = pd.Series(y, name="y_true")
    class_df = y_ser.value_counts().rename_axis("class").reset_index(name="count")
    class_df["percent"] = (class_df["count"] / class_df["count"].sum()) * 100
    class_df.to_csv(OUT_CLASS_CSV, index=False)
    print(f"Distribuição de classes salva em: {OUT_CLASS_CSV}")

    # plots
    try:
        plot_class_distribution(y_ser, OUT_CLASS_PNG)
        print(f"Gráfico de distribuição de classes salvo em: {OUT_CLASS_PNG}")
    except Exception as e:
        print("Falha ao gerar gráfico de classes:", e)

    try:
        plot_correlation(df, OUT_CORR_PNG)
        print(f"Matriz de correlação salva em: {OUT_CORR_PNG}")
    except Exception as e:
        print("Falha ao gerar matriz de correlação:", e)

    try:
        plot_feature_histograms(df, OUT_HIST_PNG)
        print(f"Histogramas de features salvos em: {OUT_HIST_PNG}")
    except Exception as e:
        print("Falha ao gerar histogramas:", e)


if __name__ == "__main__":
    main()