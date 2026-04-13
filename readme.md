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

├── data/               # Dados brutos (raw) e processados
├── metrics/            # Resultados e scores dos modelos em formato JSON
├── models/             # Modelos treinados (.joblib)
├── pipeline/           # Scripts acionados pelo DVC para execução das fases
├── src/                # Lógica central (Limpeza, Treino, Configurações)
├── dvc.yaml            # Definição das etapas do pipeline
├── params.yaml         # Centralização de hiperparâmetros e configurações
└── requirements.txt    # Dependências do projeto

Instruções de Execução

1. Configuração do Ambiente
Criação do ambiente virtual e instalação das bibliotecas necessárias:

Bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente (Windows)
.\.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
2. Execução do Pipeline
Para executar todo o fluxo de processamento e treinamento definido no DVC:

Bash
dvc repro
O DVC gerenciará as dependências entre os scripts e garantirá que o estado final seja alcançado.

3. Análise de Resultados
Para visualizar as métricas finais no terminal:

Bash
dvc metrics show
Para acessar a interface gráfica do MLflow e comparar os treinamentos realizados:

Bash
mlflow ui
Após o comando, a interface estará disponível em http://localhost:5000.

Estratégia de Modelo
O projeto utiliza um limiar de decisão (threshold) ajustado para priorizar a captura de transações fraudulentas. Esta abordagem conservadora visa reduzir o risco financeiro, aceitando um volume controlado de falsos positivos em troca de uma cobertura maior de fraudes reais.

Próximos Passos
Implementação de novas variáveis (Feature Engineering) baseadas no comportamento temporal das transações.

Adição de testes unitários para validação da integridade dos dados processados.

Integração com repositórios de armazenamento remoto para os artefatos do DVC.
