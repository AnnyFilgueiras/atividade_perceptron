# Atividade Prática — Implementação de uma rede Perceptron

**Alunos:** Anny Karoliny Germano Filgueiras e Gabryel Adriano Borges de Souza

## 1. Registro da Execução

```text
--- Iniciando treinamento ---
--- Treinamento concluído ---
Épocas executadas: 71
Erros na última época: 0
Peso final da temperatura: 0.5500
Peso final da umidade: 0.2650
Viés final: -0.6000

--- Conjunto de testes (Previsto vs. Esperado) ---
ID  1 | Temp: 23°C | Umid: 55% | Esperado: NORMAL (0) | Previsto: NORMAL (0) [OK]
ID  2 | Temp: 30°C | Umid: 50% | Esperado: NORMAL (0) | Previsto: NORMAL (0) [OK]
ID  3 | Temp: 34°C | Umid: 35% | Esperado: NORMAL (0) | Previsto: NORMAL (0) [OK]
ID  4 | Temp: 28°C | Umid: 85% | Esperado: ALERTA (1) | Previsto: ALERTA (1) [OK]
ID  5 | Temp: 33°C | Umid: 60% | Esperado: ALERTA (1) | Previsto: ALERTA (1) [OK]
ID  6 | Temp: 36°C | Umid: 70% | Esperado: ALERTA (1) | Previsto: ALERTA (1) [OK]
ID  7 | Temp: 26°C | Umid: 85% | Esperado: NORMAL (0) | Previsto: NORMAL (0) [OK]
ID  8 | Temp: 32°C | Umid: 75% | Esperado: ALERTA (1) | Previsto: ALERTA (1) [OK]

Acertos: 8/8
Acurácia no conjunto de teste: 100.00%

--- Classificação de novos valores (digite 'sair' para encerrar) ---
Digite a temperatura (°C): 33
Digite a umidade relativa (%): 80
Classificação prevista: ALERTA (1)
```

---

## 2. Respostas Curtas

### (a) Para que serve o viés (bias)?
O **viés** (*bias*) permite deslocar a fronteira de decisão (hiperplano/reta de separação) para que ela não seja obrigada a passar pela origem $(0, 0)$. Dessa forma, o neurônio consegue ajustar o limiar de ativação independentemente dos valores das entradas.

### (b) Por que normalizar as entradas?
A **normalização** coloca variáveis de grandezas diferentes (como temperatura em °C e umidade em %) em escalas próximas (neste caso, próximas ao intervalo $[0, 1]$). Isso evita que entradas com valores numéricos maiores dominem as atualizações dos pesos, tornando o treinamento mais estável e convergente em menos épocas.

> **Nota Técnica — Escala Fixa de Domínio vs. Min-Max Scaling (Sem *Data Leakage*):**

> O roteiro da atividade define uma normalização por constantes de domínio (`temperatura / 40` e `umidade / 100`), que mantém os dados de treino concentrados nos intervalos $[0.60, 0.925]$ para temperatura e $[0.30, 0.90]$ para umidade, convergindo em **71 épocas**.

> Caso utilizemos a normalização **Min-Max** ($\frac{x - x_{\min}}{x_{\max} - x_{\min}}$) ajustada **exclusivamente sobre o conjunto de treino** (evitando *Data Leakage* do conjunto de teste), ambas as variáveis passam a ocupar integralmente o intervalo $[0.0, 1.0]$. Nesse cenário (executável via `python3 main.py --minmax`), o Perceptron converge em apenas **11 épocas** (cerca de **6,4x mais rápido**) e mantém **100% de acurácia** nos testes.

### (c) O que um Perceptron simples não consegue classificar?
Um Perceptron de camada única só consegue resolver problemas **linearmente separáveis** (onde uma única reta ou hiperplano divide as duas classes). Ele **não consegue** classificar conjuntos de dados **não linearmente separáveis**, sendo o exemplo clássico a função lógica **XOR** (OU exclusivo).
