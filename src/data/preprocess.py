import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df): # Realiza a limpeza, codificação, divisão e padronização dos dados.
    columns_to_drop = ['id']
    if 'Unnamed: 32' in df.columns:
        columns_to_drop.append('Unnamed: 32')

    df_processed = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        errors='ignore'
    )

    # Codificação do Target ('B' = 0 para Benigno, 'M' = 1 para Maligno)
    if 'diagnosis' in df_processed.columns and df_processed['diagnosis'].dtype == object:
        label_map = {'B': 0, 'M': 1}
        df_processed['diagnosis'] = df_processed['diagnosis'].map(label_map)
        print("Target 'diagnosis' mapeado de B/M para 0/1.")


    # Separação entre Features (X) e Target (y)
    X = df_processed.drop('diagnosis', axis=1)
    y = df_processed['diagnosis']

    # Divisão treino/teste (20% para teste) usando stratify para manter a proporção das classes.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Padronização dos Features (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reconstituir DataFrames com nomes de colunas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Relatório
    print("\nPré-processamento concluído.")
    print(f"Total de instâncias (antes da divisão): {len(df)}")
    print(f"Treino: {len(X_train_scaled)} instâncias")
    print(f"Teste : {len(X_test_scaled)} instâncias")
    print("\nDistribuição das classes (treino):")
    print(y_train.value_counts(normalize=True).rename('proporção'))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler