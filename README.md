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
