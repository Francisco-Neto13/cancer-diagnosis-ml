# 🩺📊 Avaliação de Modelos de Aprendizado de Máquina Aplicados ao Diagnóstico de Câncer de Mama

### 🧬📊 Pipeline Completa de Diagnóstico — Fluxo, Arquitetura e Funções dos Arquivos

## 👥 Participantes  
- *Caio Gabriel Pereira de Menezes Correia*  
- *Caio Renatos dos Santos Claudino*  
- *Jose Francisco de Araújo Neto*  
- *Lucas Zonzini Lisboa*  

---

## 📖 Descrição Geral  
Este documento descreve o fluxo de execução do projeto, explicando o papel de cada arquivo Python e como eles se conectam.  
O **arquivo central de controle é o `main.py`**, responsável por orquestrar toda a pipeline.

---

# 📂 Ordem de Execução e Função de Cada Arquivo

## 1️⃣ main.py — *Arquivo Principal*
Função principal: ponto de entrada da pipeline.  
Responsabilidades:
- Configura o ambiente inicial.
- Cria as pastas **`models/`** e **`results/`** caso não existam.
- Encadeia a execução das fases:
  - Dados
  - Treinamento
  - Avaliação
  - Análise de erros

  ---

## 2️⃣ FASE DE DADOS E PRÉ-PROCESSAMENTO

### 2.1. loader.py
**Função:**  
Carrega os dados brutos, seja a partir de:
- arquivos CSV
- datasets nativos do *scikit-learn*

---

## 2.2. preprocess.py
**Função:**  
Executa as transformações essenciais no dataset.

**Ações realizadas:**
- Remoção de colunas irrelevantes
- Codificação do target (B/M → 0/1)
- Divisão treinamento/teste
- Padronização das features com **StandardScaler**

---

## 2.3. describe_dataset.py
**Função:**  
Realiza Análise Exploratória de Dados (EDA).

**Ações:**
- Gera estatísticas descritivas
- Cria a matriz de correlação
- Plota gráficos de distribuição e histogramas  
➡️ Todos os resultados são salvos na pasta **`results/`**

---

## 3️⃣ FASE DE MODELOS E TREINAMENTO

## 3.1. model_hyperparameters.py
**Função:**  
Define os modelos utilizados no experimento e seus hiperparâmetros.

Inclui:
- Modelos base (Ex: LogisticRegression, KNN)
- Grades de hiperparâmetros para otimização futura

---

## 3.2. train.py
**Função:**  
Responsável por orquestrar todo o treinamento.

**Ações:**
- Carrega e pré-processa os dados
- Treina todos os modelos definidos em `model_hyperparameters.py`
- Salva os artefatos de ML:
  - modelos treinados
  - scaler
  - test_data  
➡️ Tudo salvo dentro da pasta **`models/`**

---

## 3.3. model_utils.py
**Função:**  
Funções utilitárias para carregar modelos treinados (`*.joblib`)  
Usado nas fases de avaliação e análise.

---

# 4️⃣ FASE DE AVALIAÇÃO E ANÁLISE

## 4.1. evaluate.py
**Função:**  
Avaliar o desempenho dos modelos no conjunto de teste.

**Ações:**
- Calcula métricas:
  - F1-score
  - AUC
  - Accuracy
- Gera uma tabela comparativa em CSV  
- Plota a **Curva ROC comparativa** entre os modelos

---

## 4.2. predict_and_visualize.py
**Função:**  
Executa previsões finais de todos os modelos.

**Ações:**
- Cria a tabela consolidada:  
  **`predictions_table.csv`**
- Esta tabela é usada na análise de erros

---

## 4.3. error_analysis_table.py
**Função:**  
Gera a análise detalhada de erros (FP e FN).

**Ações:**
- Calcula resumo de falsos positivos e falsos negativos
- Lista exemplos de erro
- Cria gráficos de erro  
➡️ Tudo salvo dentro de **`results/`**

---

## 4.4. feature_importance.py
**Função:**  
Analisa e visualiza a importância das features  
(Apenas para modelos que suportam esse cálculo)

---

# 🔄 Fluxo Resumido do Projeto

1. **loader.py**  
   ⤷ Carrega os dados brutos

2. **preprocess.py**  
   ⤷ Limpa, padroniza e cria `X_train`, `X_test`, `y_train`, `y_test`

3. **train.py**  
   ⤷ Treina os modelos e salva os artefatos em **models/**

4. **describe_dataset.py**  
   ⤷ Gera análise exploratória sobre o conjunto de teste

5. **evaluate.py**  
   ⤷ Calcula métricas (F1, AUC, Accuracy) e produz as curvas ROC

6. **predict_and_visualize.py**  
   ⤷ Gera `predictions_table.csv` com as previsões de todos os modelos

7. **error_analysis_table.py**  
   ⤷ Produz a análise de erros (FP vs FN) e gráficos consolidados

---

# 📝 Observações Importantes

- A pasta **`models/`** armazena artefatos de Machine Learning.  
- A pasta **`results/`** armazena relatórios, tabelas e gráficos.  
- **Ambas devem ser ignoradas no Git** (`.gitignore`).  
---
