Fraud Detection Pipeline (MLOps)

Este projeto implementa um sistema de detecção de fraudes em pagamentos digitais utilizando técnicas modernas de MLOps para garantir a reprodutibilidade, o versionamento de dados e o rastreamento de experimentos.

Diferenciais do Projeto

Diferente de um modelo comum em Jupyter Notebook, este projeto utiliza:

DVC (Data Version Control): Gerenciamento do pipeline de dados e garantia de reprodutibilidade.

MLflow: Rastreamento de experimentos, métricas e versionamento de modelos.

Estrutura Modular: Código organizado em pacotes (src) e scripts de pipeline, pronto para produção.

Tecnologias Utilizadas
Linguagem: Python 3.14.4

Machine Learning: XGBoost, Scikit-Learn (Random Forest)

Orquestração: DVC

Tracking: MLflow

Ambiente: Venv (Virtual Environment)

Estrutura do Repositório
Plaintext
├── data/               # Dados brutos (raw) e processados
├── metrics/            # Resultados e scores dos modelos
├── models/             # Modelos treinados (.joblib)
├── pipeline/           # Scripts de execução das etapas do DVC
├── src/                # Lógica central (Limpeza, Treino, Configurações)
├── dvc.yaml            # Configuração do pipeline de dados
├── params.yaml         # Hiperparâmetros centralizados
└── requirements.txt    # Dependências do projeto
Como Executar
1. Configurar o Ambiente

# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows)
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
2. Rodar o Pipeline Completo
Graças ao DVC, você não precisa rodar script por script. O comando abaixo executa todo o fluxo (limpeza -> treino -> avaliação):

dvc repro

3. Visualizar Métricas
Para ver os resultados no terminal:

dvc metrics show

Para abrir a interface visual do MLflow e comparar os modelos graficamente:

mlflow ui
(Após rodar, acesse http://localhost:5000 no seu navegador)

Resultados Atuais
O projeto compara dois algoritmos principais: XGBoost e Random Forest. O foco atual é a detecção agressiva de fraudes, priorizando a captura de casos suspeitos através de um threshold ajustado.

Próximos Passos
1 - Realizar Feature Engineering avançada para melhorar o F1-Score.

2 - Implementar testes unitários para a etapa de limpeza.

3 - Configurar um servidor remoto para o MLflow.