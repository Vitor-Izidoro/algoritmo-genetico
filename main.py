import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time



# Configuração do Problema


RANDOM_STATE = 42
POP_SIZE = 20          # Tamanho da população
N_GENERATIONS = 30     # Número máximo de gerações
MUTATION_RATE = 0.05   # Probabilidade de mutação
ELITISM = True         # Manter melhor indivíduo



# Carrega e dividi os dados


data = load_breast_cancer()
X = data.data
y = data.target

# 60% treino, 20% validação, 20% teste
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE
)

N_FEATURES = X.shape[1]



# Funções auxiliares


def initialize_population():
    #Cria uma população inicial binária aleatória
    return np.random.randint(2, size=(POP_SIZE, N_FEATURES))


def evaluate_individual(individual):
    #Fitness = acurácia do KNN + pequena penalização pelo nº de atributos
    selected = np.where(individual == 1)[0]
    
    if len(selected) == 0:
        return 0  # Caso especial: cromossomo sem features não é válido
    
    knn = KNeighborsClassifier()
    knn.fit(X_train[:, selected], y_train)
    y_pred = knn.predict(X_val[:, selected])
    acc = accuracy_score(y_val, y_pred)
    
    # Penalização opcional para reduzir número de features
    penalty = len(selected) / N_FEATURES
    
    return acc - penalty * 0.01  # Ajuste conforme desejar


def evaluate_population(population):
    #Retorna fitness de todos os indivíduos
    return np.array([evaluate_individual(ind) for ind in population])


def selection(population, fitness):
    #Seleção por Roleta
    probs = fitness / np.sum(fitness)
    parents_idx = np.random.choice(len(population), size=2, p=probs, replace=False)
    return population[parents_idx[0]], population[parents_idx[1]]


def crossover(parent1, parent2):
    #Crossover de um ponto
    point = random.randint(1, N_FEATURES - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


def mutate(individual):
    #Mutação bit-flip
    for i in range(N_FEATURES):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual


def get_best(population, fitness):
    idx = np.argmax(fitness)
    return population[idx], fitness[idx]


def evaluate_all_features():
    #Avalia o classificador KNN utilizando todas as features.
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return acc


# Loop do Algoritmo Genético

def genetic_algorithm():
    population = initialize_population()
    best_global_ind, best_global_fit = None, -np.inf
    start_time = time.time()

    for gen in range(N_GENERATIONS):
        fitness = evaluate_population(population)
        new_population = []

        # Elitismo
        if ELITISM:
            best_ind, best_fit = get_best(population, fitness)
            new_population.append(best_ind.copy())

        # Gerar nova população
        while len(new_population) < POP_SIZE:
            p1, p2 = selection(population, fitness)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(c2))

        population = np.array(new_population)

        # Melhor da geração
        gen_best_ind, gen_best_fit = get_best(population, evaluate_population(population))

        if gen_best_fit > best_global_fit:
            best_global_ind = gen_best_ind.copy()
            best_global_fit = gen_best_fit

        print(f"Geração {gen+1}/{N_GENERATIONS} | Melhor Fitness: {best_global_fit:.4f}")

    execution_time = time.time() - start_time
    return best_global_ind, best_global_fit, execution_time


# Avaliação com todas as features
print("\n Avaliando classificador com todas as features...")
all_features_acc = evaluate_all_features()
print(f"Acurácia (todas as features): {all_features_acc:.4f}")

# Executar Algoritmo Genético
print("\n Executando Algoritmo Genético...")
best_individual, best_fitness, ga_execution_time = genetic_algorithm()

# Avaliar melhor indivíduo no conjunto de teste
selected_features = np.where(best_individual == 1)[0]
knn = KNeighborsClassifier()
knn.fit(X_train[:, selected_features], y_train)
y_test_pred = knn.predict(X_test[:, selected_features])
test_accuracy = accuracy_score(y_test, y_test_pred)

# Resultados finais
print("\n Resultados Finais:")
print(f"Acurácia (todas as features): {all_features_acc:.4f}")
print(f"Acurácia (GA - conjunto de teste): {test_accuracy:.4f}")
print(f"Tempo de execução (GA): {ga_execution_time:.2f} segundos")
print(f"Nº Features Selecionadas: {np.sum(best_individual)}")

import pandas as pd

# Dados para a tabela
results = {
    "Feature Selection Method": ["Without Selection", "GA (Genetic Algorithm)"],
    "# of Features": [30, np.sum(best_individual)],
    "Accuracy Test Set (%)": [all_features_acc * 100, test_accuracy * 100],
    "Execution time (s)": [None, ga_execution_time]
}

# Criar DataFrame
df = pd.DataFrame(results)

# Salvar como CSV
df.to_csv("comparative_results.csv", index=False)

# Exibir tabela
print("\nTabela de Resultados Comparativos:")
print(df)
