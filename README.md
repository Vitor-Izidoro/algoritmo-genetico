# Otimização de Seleção de Atributos com Algoritmo Genético

##  Objetivo
Aplicar técnicas de computação evolutiva para otimizar a seleção de atributos.  
O objetivo é melhorar a acurácia do classificador enquanto se reduz a dimensionalidade dos atributos, utilizando uma estratégia baseada em *wrapper* guiada por um Algoritmo Genético (AG).

---

##  Descrição do Projeto
Cada instância do dataset **Breast Cancer** possui **30 atributos**.  
Os alunos (em grupos de até 4 pessoas) devem implementar um **Algoritmo Genético** para buscar o subconjunto ótimo de atributos que maximize a acurácia do classificador.

O subconjunto de atributos selecionado será avaliado utilizando o **KNN do Scikit-Learn**, com parâmetros padrão.

O conjunto de dados deverá ser dividido em:
- **60%** para treinamento
- **20%** para validação
- **20%** para teste

Em cada geração do AG:
- Cada indivíduo representa um subconjunto de atributos
- O classificador será treinado no conjunto de **treinamento**
- A avaliação será realizada no conjunto de **validação**

Além disso, os resultados obtidos devem ser comparados com o desempenho de:
-  Classificador utilizando **todos os atributos**

---

##  Explicação do Código

O código implementa um **Algoritmo Genético (AG)** para otimizar a seleção de atributos no dataset **Breast Cancer**. Aqui está um resumo das etapas principais:

1. **Carregamento e Divisão dos Dados**:
   - O dataset é dividido em 60% para treinamento, 20% para validação e 20% para teste.

2. **Inicialização da População**:
   - Cada indivíduo na população é representado por um vetor binário, onde cada posição indica se uma feature está selecionada (1) ou não (0).

3. **Avaliação de Fitness**:
   - O fitness de cada indivíduo é calculado com base na acurácia do classificador KNN no conjunto de validação, com uma penalização para subconjuntos com muitas features.

4. **Operadores Genéticos**:
   - **Seleção**: Utiliza o método de roleta para selecionar os pais.
   - **Crossover**: Combina genes de dois pais para criar novos indivíduos.
   - **Mutação**: Altera aleatoriamente os genes de um indivíduo com uma pequena probabilidade.

5. **Elitismo**:
   - O melhor indivíduo de cada geração é mantido na próxima geração para garantir que o desempenho não piore.

6. **Comparação com Classificador Completo**:
   - O desempenho do AG é comparado com o classificador KNN utilizando todas as features.

7. **Resultados**:
   - O código exibe a acurácia do classificador com todas as features, a acurácia do melhor indivíduo no conjunto de teste, o número de features selecionadas e o tempo de execução do AG.

---

##  Entregáveis
 **Relatório em PDF**, contendo:
- Descrição dos parâmetros e implementação do Algoritmo Genético
- Resultados comparativos apresentados em tabela conforme modelo solicitado

 **Código-fonte (Python)**

---

##  Tecnologias e Bibliotecas Recomendadas
- Python 3.x
- Scikit-learn
- Numpy
- Pandas
- (Opcional) Matplotlib / Seaborn para visualizações

---

##  Autores
- `Gabriel simini, Murilo Pedrazzani, Vitor IZidoro`

---

##  Dataset Utilizado
 Disponível no Scikit-Learn:
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```
