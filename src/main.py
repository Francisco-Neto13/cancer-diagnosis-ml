import os
import sys

sys.path.append(os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

from data.describe_dataset import run_dataset_description
from data.loader import load_raw_data

from models.train import run_training
from models.model_utils import load_trained_models 

from evaluation.evaluate import run_evaluation
from evaluation.error_analysis_table import run_error_analysis

from analysis.predict_and_visualize import run_prediction_analysis
from analysis.feature_importance import run_feature_importance_analysis


MODEL_NAMES = ['LogisticRegression', 'KNN', 'SVC_RBF', 'RandomForest']


def setup_project(): # Cria os diretórios 'models/', 'results/' e 'data/' na raiz do projeto.
    print("## INICIANDO CONFIGURAÇÃO DO PROJETO ##")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Diretório '{os.path.basename(MODELS_DIR)}/' verificado/criado em: {MODELS_DIR}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Diretório '{os.path.basename(RESULTS_DIR)}/' verificado/criado em: {RESULTS_DIR}")
    
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True) 
    
    print("------------------------------------------")


def main_pipeline(step="all"): # Orquestra o fluxo completo de ML com base no passo especificado ('train', 'evaluate', 'analyse', 'describe' ou 'all').
    setup_project()
    
    # DADOS E PRÉ-PROCESSAMENTO
    if step in ["all", "train", "describe"]:
        print("\n\n=============== FASE 1: DADOS E PRÉ-PROCESSAMENTO ===============")
        try:
            load_raw_data() 
            
            run_dataset_description()
            
        except Exception as e:
            print(f"ERRO FATAL na FASE 1 (Dados): {e}")
            return


    # TREINAMENTO DOS MODELOS
    if step in ["all", "train"]:
        print("\n\n===================== FASE 2: TREINAMENTO =======================")
        try:
            run_training() 
        except Exception as e:
            print(f"ERRO FATAL na FASE 2 (Treinamento): {e}")
            return
            
            
    # AVALIAÇÃO E ANÁLISE
    if step in ["all", "evaluate", "analyse"]:
        print("\n\n===================== FASE 3: AVALIAÇÃO =========================")
        
        try:
            print("Carregando modelos treinados do disco...")
            trained_models = load_trained_models(MODEL_NAMES) 
            
            if not trained_models:
                print("ERRO FATAL: Nenhum modelo carregado. Execute 'python src/main.py train' primeiro.")
                return
                
        except Exception as e:
            print(f"ERRO FATAL ao carregar modelos para avaliação: {e}")
            return

        try:
            # Avaliação: Métricas, Curvas ROC, Matrizes de Confusão
            run_evaluation(MODEL_NAMES) 
        except Exception as e:
            print(f"ERRO na Avaliação: {e}. Prosseguindo com a Análise...")


        print("\n\n====================== FASE 4: ANÁLISE =========================")
        try:
            # Tabela de Previsões e Gráfico de Acurácia (Gera predictions_table.csv)
            run_prediction_analysis()
            
            # Análise de Erros (FP vs FN) - Requer predictions_table.csv
            run_error_analysis()
            
            # Importância das Features
            run_feature_importance_analysis()
            
        except Exception as e:
            print(f"ERRO na Análise: {e}")
            
    print("\n==================  PIPELINE CONCLUÍDA  =====================")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["train", "evaluate", "analyse", "describe"]:
        main_pipeline(sys.argv[1])
    else:
        main_pipeline("all")