# 🔍 Detecção de Anomalias em Séries Temporais com LSTM Autoencoder

Este projeto implementa um sistema de detecção de anomalias em dados de sensores industriais (temperatura, vibração, etc.) utilizando um **autoencoder LSTM** e compara seus resultados com métodos estatísticos clássicos (Z-Score, IQR, Média Móvel).

## 📌 Visão Geral

- **Objetivo:** Identificar padrões anômalos em séries temporais de sensores que podem indicar falhas, eventos incomuns ou mudanças no comportamento do sistema.
- **Abordagem:** Um autoencoder baseado em LSTM aprende a reconstruir sequências normais de dados; sequências com alto erro de reconstrução são classificadas como anomalias.
- **Diferenciais:** 
  - Uso de **RobustScaler** para minimizar o impacto de outliers extremos.
  - **Limiar adaptativo** baseado na evolução do erro de reconstrução.
  - Comparação visual e numérica com métodos tradicionais.

## 📊 Resultados

A imagem abaixo mostra a comparação entre os métodos aplicados a um dataset de temperatura ambiente:

![Comparação de Métodos de Detecção de Anomalias](imagem_comparacao.png)

### Estatísticas comparativas

| Método               | Anomalias Detectadas | Percentual |
|----------------------|----------------------|------------|
| Z-Score (3σ)         | 16                   | 0,32%      |
| IQR                  | 42                   | 0,84%      |
| Média Móvel          | 6                    | 0,12%      |
| **LSTM Autoencoder** | **67**               | **1,34%**  |

O LSTM Autoencoder foi capaz de capturar **padrões não lineares e sutis** que os métodos clássicos não identificaram, resultando em uma detecção mais abrangente.

## 🧠 Modelo LSTM

Arquitetura utilizada:

```python
model = keras.Sequential([
    layers.Input(shape=(WINDOW, 1)),
    layers.LSTM(128, return_sequences=True, dropout=0.2),
    layers.LSTM(64, return_sequences=False, dropout=0.2),
    layers.RepeatVector(WINDOW),
    layers.LSTM(64, return_sequences=True, dropout=0.2),
    layers.LSTM(128, return_sequences=True, dropout=0.2),
    layers.TimeDistributed(layers.Dense(1))
])
