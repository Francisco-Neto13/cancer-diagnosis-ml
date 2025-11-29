import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


def load_data(file_path='data/breast_cancer.csv'): # Função para carregar o dataset e verificar existência do arquivo 
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset carregado de: {file_path}")
        return df

    except FileNotFoundError:
        print(f"Arquivo não encontrado em {file_path}.")
        print("Carregando dataset nativo do scikit-learn...")

        cancer = load_breast_cancer(as_frame=True)
        df = cancer.frame

        # Ajuste de compatibilidade
        df.rename(columns={'target': 'diagnosis'}, inplace=True)

        print("Dataset carregado com sucesso do scikit-learn.")
        return df


def preprocess_data(df): # Função de pré-processamento completo do dataset. Removendo colunas, separação de feat e target, padronização, etc
    # 1. Remoção de colunas desnecessárias
    columns_to_drop = ['id']
    if 'Unnamed: 32' in df.columns:
        columns_to_drop.append('Unnamed: 32')

    df_processed = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        errors='ignore'
    )

    # 2. Converter labels para 0/1
    label_map = {'B': 0, 'M': 1}
    
    # Caso esteja usando o dataset nativo do sklearn (0/1), ignora
    if df_processed['diagnosis'].dtype == object:
        df_processed['diagnosis'] = df_processed['diagnosis'].map(label_map)

    # 3. Separação entre X e y
    X = df_processed.drop('diagnosis', axis=1)
    y = df_processed['diagnosis']

    # 4. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Relatório
    print("\nPré-processamento concluído.")
    print(f"Total de instâncias: {len(df)}")
    print(f"Treino: {len(X_train_scaled)} instâncias")
    print(f"Teste : {len(X_test_scaled)} instâncias")
    print("\nDistribuição das classes (treino):")
    print(y_train.value_counts(normalize=True).rename('proporção'))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

