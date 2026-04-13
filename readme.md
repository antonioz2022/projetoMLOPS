# Fraud Detection Pipeline (MLOps)

Este projeto implementa um sistema de detecção de fraudes em pagamentos digitais utilizando técnicas modernas de MLOps. O foco principal é garantir a reprodutibilidade do experimento, o versionamento de dados e o rastreamento automatizado de métricas.

## Diferenciais do Projeto

Diferente de abordagens baseadas exclusivamente em notebooks, este projeto utiliza uma estrutura de engenharia de software:

* **DVC (Data Version Control):** Orquestração do pipeline de dados. O sistema identifica alterações em dados ou scripts e executa apenas as etapas necessárias.
* **MLflow:** Rastreamento de experimentos, registro de hiperparâmetros e comparação visual de performance entre modelos.
* **Estrutura Modular:** Código organizado em módulos Python (pasta `src`) e scripts de execução de pipeline, facilitando a manutenção e escala.


## Tecnologias Utilizadas

* **Linguagem:** Python 3.12+
* **Machine Learning:** XGBoost e Scikit-Learn (Random Forest)
* **Orquestração de Pipeline:** DVC
* **Rastreamento de Experimentos:** MLflow
* **Gerenciamento de Ambiente:** Venv (Virtual Environment)


## Estrutura do Repositório

text
├── data/               # Dados brutos (raw) e processados
├── metrics/            # Resultados e scores dos modelos em formato JSON
├── models/             # Modelos treinados (.joblib)
├── pipeline/           # Scripts acionados pelo DVC para execução das fases
├── src/                # Lógica central (Limpeza, Treino, Configurações)
├── dvc.yaml            # Definição das etapas do pipeline
├── params.yaml         # Centralização de hiperparâmetros e configurações
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

'dvc repro'

3. Visualizar Métricas
Para ver os resultados no terminal:

'dvc metrics show'

Para abrir a interface visual do MLflow e comparar os modelos graficamente:

'mlflow ui'
(Após rodar, acesse http://localhost:5000 no seu navegador)

Resultados Atuais
O projeto compara dois algoritmos principais: XGBoost e Random Forest. O foco atual é a detecção agressiva de fraudes, priorizando a captura de casos suspeitos através de um threshold ajustado.

Próximos Passos
1 - Realizar Feature Engineering avançada para melhorar o F1-Score.

2 - Implementar testes unitários para a etapa de limpeza.

3 - Configurar um servidor remoto para o MLflow.
