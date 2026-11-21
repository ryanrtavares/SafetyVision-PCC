# SafetyVision: Monitoramento de EPIs e Segurança Industrial em Tempo Real com YOLO11m

O SafetyVision é uma solução de visão computacional baseada em Aprendizado Profundo desenvolvida para a detecção automática de Equipamentos de Proteção Individual (EPIs), sinalização de não-conformidades de segurança e identificação de eventos críticos em ambientes industriais e canteiros de obras.

Este repositório é parte integrante do Projeto de Conclusão de Curso (PCC) do curso de Ciência de Dados e Business Intelligence da Universidade Católica de Petrópolis (UCP), em conformidade com o Regulamento de PCC (Art. 3º, § 1º).

---

## Destaques e Resultados Principais

- Modelo: YOLO11m (Ultralytics) com precisão mista (FP16).
- Dataset: Mapeamento semântico de 14 classes (EPIs, infrações e elementos de cenário).
- mAP@0.5: 79,58% (pico na Época 86) / 79,36% (Época 98).
- mAP@0.5:0.95: 52,07%.
- Recall (Sensibilidade): 85,05% (foco em minimizar falsos negativos na segurança).
- F1-Score: 77,95%.
- Inferência em Tempo Real: Captura e processamento via OpenCV com integração ao OBS Studio.

---

## REQUISITO CRÍTICO: Treinamento na Plataforma Kaggle

> ATENÇÃO / IMPORTANTE: O script de treinamento (scripts/train_safetyvision.py) foi projetado e otimizado especificamente para ser executado no ambiente Kaggle Notebooks com aceleração por GPU (NVIDIA T4 ou P100).

### Por que o treinamento deve ser executado no Kaggle?
1. Poder de Processamento: A arquitetura YOLO11m e a resolução de imagem de 960x960 exigem aceleração por GPU dedicada com suporte a FP16 (Half-Precision) para completar as 98 épocas em tempo hábil.
2. Estrutura de Diretórios Graváveis: No Kaggle, o dataset e os arquivos compactados devem ser manipulados no diretório /kaggle/working/ para evitar erros de permissão de escrita durante a geração de relatórios e salvamento dos checkpoints (best.pt e last.pt).
3. Gestão de Sessão e Checkpoints: Devido ao limite de tempo de execução contínua da sessão do Kaggle (ex.: limite de 12 horas), o treinamento utiliza salvamento automático por época, permitindo a retomada transparente via parâmetro resume=True a partir do checkpoint last.pt.

---

## Estrutura do Repositório

```text
SafetyVision-PCC/
├── README.md                     <- Documentação principal do projeto
├── requirements.txt              <- Lista de dependências Python
├── data/
│   ├── data.yaml                 <- Arquivo de configuração das 14 classes
│   └── result11m9.csv            <- Histórico contínuo do treinamento (98 épocas)
├── scripts/
│   ├── train_safetyvision.py     <- Script de treinamento do YOLO11m (Kaggle GPU)
│   ├── generate_metrics_plots.py <- Script dos gráficos de métricas globais e loss
│   ├── generate_confusion_matrix.py <- Script da matriz de confusão (14 classes)
│   └── inference_obs_realtime.py <- Script de inferência em tempo real via OBS Studio
└── figures/                      <- Gráficos e capturas em alta resolução
    ├── grafico_curvas_aprendizado_metricas.png
    ├── grafico_perdas_treino_validacao.png
    ├── grafico_tradeoff_f1_lr.png
    └── matriz_confusao_safetyvision_14classes.png
```

---

## Classes Mapeadas (14 Variáveis Semânticas)

O dataset abrange 14 classes divididas em 3 categorias operacionais:

1. EPIs em Conformidade: Person, Hardhat, Safety Vest, Mask, Goggles, Gloves.
2. Infrações de Segurança: NO-Hardhat, NO-Safety Vest, NO-Mask, NO-Goggles, NO-Gloves.
3. Cenário & Eventos Críticos: Safety Cone, Ladder, Fall-Detected.

---

## Como Executar

### 1. Treinamento do Modelo (No Kaggle)
1. Crie um novo Notebook na plataforma Kaggle.
2. Ative a aceleração de hardware: Settings -> Accelerator -> GPU T4 x2 (ou P100).
3. Importe o dataset de imagens e o arquivo data.yaml.
4. Execute o script scripts/train_safetyvision.py:
   ```bash
   python scripts/train_safetyvision.py
   ```
5. Ao finalizar o ciclo, faça o download dos pesos otimizados best.pt.

### 2. Geração dos Gráficos de Resultados (Localmente ou no Colab/Kaggle)
Para replicar os gráficos oficiais apresentados no relatório científico a partir de result11m9.csv:
```bash
pip install -r requirements.txt
python scripts/generate_metrics_plots.py
python scripts/generate_confusion_matrix.py
```

### 3. Inferência Visual em Tempo Real (Localmente)
Para executar a detecção contínua utilizando webcam ou câmera virtual do OBS Studio:
```bash
# Baixe o arquivo best.pt para a raiz do repositório
python scripts/inference_obs_realtime.py
```

---

## Autoria e Instituição

- Autor: Ryan Ramaldis Tavares
- Curso: Bacharelado em Ciência de Dados e Business Intelligence
- Instituição: Universidade Católica de Petrópolis (UCP)
- Ano: 2026
