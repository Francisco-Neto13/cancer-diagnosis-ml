# 🩺📊 Avaliação de Modelos de Aprendizado de Máquina Aplicados ao Diagnóstico de Câncer de Mama

## 👥 Participantes  
- *Caio Gabriel Pereira de Menezes Correia*  
- *Caio Renatos dos Santos Claudino*  
- *Jose Francisco de Araújo Neto*  
- *Lucas Zonzini Lisboa*  

---

## 📖 Descrição Geral  
Este repositório apresenta uma pipeline completa para análise, treinamento e avaliação de modelos de Machine Learning aplicados ao diagnóstico de câncer de mama.  
O projeto é dividido em quatro partes principais: **preprocessamento**, **treinamento**, **avaliação dos modelos** e **apresentação dos resultados**.

---

# 📂 Ordem de Execução e Função de Cada Arquivo

## 1️⃣ preprocess.py  
*(Não precisa ser executado diretamente pelo usuário — incluído apenas para documentação.)*

### 🔧 Funções principais:
- Carregar o dataset original.  
- Tratar dados (remoção de valores faltantes, normalização e codificação de variáveis categóricas).  
- Realizar a divisão entre treino e teste.  
- Retornar:  'X_train', 'X_test', 'y_train',y_test

  ---

## 2️⃣ train.py

### 🔧 Função principal:
- Receber `X_train` e `y_train` processados pelo `preprocess.py`.  
- Treinar múltiplos modelos, como:  
- Logistic Regression  
- KNN  
- SVC (RBF)  
- Random Forest  
- Salvar os modelos treinados na pasta `models/` (criada automaticamente).  
- Retornar métricas básicas de desempenho nos dados de treinamento.

---

## 3️⃣ evaluate.py

### 🔧 Função principal:
- Carregar os modelos treinados salvos em `models/`.  
- Avaliar usando `X_test` e `y_test`.  
- Gerar métricas como:  
- AUC  
- Accuracy  
- Matriz de Confusão  
- Criar visualizações, como:  
- Curvas ROC comparativas  
- Exibir ou salvar os resultados obtidos.

---

# 🔄 Fluxo Resumido do Projeto

preprocess.py
→ limpa e prepara os dados (normalização, codificação, divisão)

train.py
→ treina os modelos com X_train, y_train
→ salva os modelos em "models/"

evaluate.py
→ carrega os modelos
→ avalia com X_test, y_test
→ gera as métricas e gráficos

---

# 📝 Observações Importantes

- A pasta **`models/`** é gerada automaticamente e **não deve ser versionada no Git**.  
- A pasta **`__pycache__/`** é criada automaticamente pelo Python e também deve ser ignorada.  
- A ordem lógica de execução deve ser respeitada:  
  1. `preprocess.py`  
  2. `train.py`  
  3. `evaluate.py`  

---
