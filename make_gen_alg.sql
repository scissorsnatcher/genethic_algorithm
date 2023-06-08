USE [CoverTable]
GO

IF OBJECT_ID ( 'Genetic_Algorithm', 'P' ) IS NOT NULL
    DROP PROCEDURE Genetic_Algorithm;
	PRINT 'DELETE';
GO


CREATE PROCEDURE Genetic_Algorithm  (
		@tableName VARCHAR(80), -- название таблицы с покрытием (например "Cover_Table")
		@pop_size INT = 50, 
		@crossover FLOAT = 1.0,
		@mutation FLOAT = 1.0,
		@max_generation INT = 1000,
		@max_generation_better INT = 500,
		@p FLOAT = 0.5
)
AS

DECLARE @requestTable NVARCHAR(max)

SET @requestTable = 'SELECT ['+@tableName+'].I, ['+@tableName+'].J FROM ['+@tableName+']';


BEGIN
	-- создать таблицу результат (просто набор номеров вошедших рядов)
	IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Genetic_Algorithm_Result')
	BEGIN
		PRINT 'NOT EXISTS'
		Create Table Genetic_Algorithm_Result (RowInCover INT);
	END
	TRUNCATE TABLE Genetic_Algorithm_Result;


	INSERT INTO Genetic_Algorithm_Result 
	EXECUTE sp_execute_external_script @language = N'Python',
	@script = N'
import random as rand
from collections import Counter
import pyodbc
from datetime import datetime
import pandas as pd
import numpy as np
import time
import warnings

warnings.simplefilter("ignore", UserWarning)

# чтобы программа успешно отрабатала,в sqlite необходимо выполнить запросы dbo.Cvt_Create и fil.Kotm_f1
# константы генетического алгоритма
POPULATION_SIZE = pop_size  # количество индивидуумов в популяции, для производительности можно понизить
P_CROSSOVER = crossover  # вероятность скрещивания
P_MUTATION = mutation # вероятность мутации индивидуума
MAX_GENERATIONS = max_generation # максимальное количество поколений, можно понизить для производительности
MAX_GENERATIONS_BETTER = max_generation_better  # максимальное количество поколений за которое решение может быть улучшено решение, можно понизить для производительности
PROBABILITY = p  # вероятность генерации единиц в исходной популяции.

class Fitness():
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = Fitness()


def fitnessUnweighted(individual):
    return sum(individual),  # кортеж

def fitnessWeighted(population, arr):
    fitness_arr = [0]*len(population)
    for i in range(len(population)):
        for j in range(len(population[i])):
            if population[i][j] == 1:
                fitness_arr[i] += sum(arr[j])
    return fitness_arr


def selTournament(population, p_len):
    offspring = []
    arr_ind = []
    for n in range(2):
        i1 = i2 = i3 = -1
        while (i1 == i2 or i1 == i3 or i2 == i3) or (i1 in arr_ind or i2 in arr_ind or i3 in arr_ind):
            i1, i2, i3 = rand.randint(0, p_len - 1), rand.randint(0, p_len - 1), rand.randint(0, p_len - 1)
        select = [[population[i1].fitness.values[0], i1], [population[i2].fitness.values[0], i2],
                  [population[i3].fitness.values[0], i3]]
        best_fitness = min(np.array(select)[:, 0])
        for i in select:
            if i[0] == best_fitness:
                arr_ind.append(i[1])
                offspring.append(population[i[1]])
                break

    for i in range(p_len):
        if i not in arr_ind:
            offspring.append(population[i])
    return offspring


def cxOnePoint(parent1, parent2):
    s = rand.randint(0, len(parent1) - 1)
    child = [] * len(parent1)
    child[:s], child[s:] = parent1[:s], parent2[s:]
    return child


def mutFlipBit(mutant, m):
    indx = rand.randint(0, m - 1)
    mutant[indx] = 0 if mutant[indx] == 1 else 1


def individualCreator(arr, n, m):
    while True:
        individ = Individual([1 if rand.randint(0, 100) < PROBABILITY * 100 else 0 for i in range(m)])
        if checkCover(individ, arr, n, m):
            return individ


def populationCreator(arr, pop_size, n, m):
    return list([individualCreator(arr, n, m) for i in range(pop_size)])


def clone(value):
    ind = Individual(value[:])
    ind.fitness.values = value.fitness.values
    return ind


def checkCover(mutant, arr, n, m):
    new_arr = list(arr)
    cover = [0] * n
    for gen_num in range(m):
        if mutant[gen_num] == 1:
            cover |= new_arr[gen_num]
    if 0 not in cover:
        return 1
    else:
        return 0


def fitnessUpdate(population, arr):
    freshFitnessValues = list(map(fitnessUnweighted, population))
    for individual, fitnessValue in zip(population, freshFitnessValues):
         individual.fitness.values = fitnessValue
    #freshFitnessValues = fitnessWeighted(population, arr)
    #for individual, fitnessValue in zip(population, freshFitnessValues):
    #   individual.fitness.values[0] = fitnessValue

def individOpt(child, arr, n, m):
    new_arr = list(arr)
    cover = [0] * n
    cover_arr = []
    new_child = [0]* m
    for gen_num in range(m):
        if child[gen_num] == 1:
            cover |= new_arr[gen_num]
            cover_arr.append([new_arr[gen_num], gen_num])

    nonCoverElemInd = []
    for i in range(len(cover)):
        if cover[i] == 0:
            nonCoverElemInd.append(i)
    ElemCover = []
    for i in range(len(new_arr)):
        for j in nonCoverElemInd:
            if new_arr[i][j] == 1:
                ElemCover.append(i)

    ElemCoverSort = Counter(ElemCover)
    freq = 10000
    for i in ElemCoverSort.most_common():
        if freq > i[1]:
            freq = i[1]
            cover_arr.append([new_arr[i[0]], i[0]])
    cover_arr = cover_arr[::-1]
    el = 0
    w = []
    for j in range(len(cover_arr[0][0])):
        w.append(sum(list(np.array(list(np.array(cover_arr)[:, 0]))[:, j])))
    while el < len(cover_arr):
        sum_el = 0
        for j in range(len(cover_arr[el][0])):
            if cover_arr[el][0][j] == 1 and w[j] > 1:
                sum_el += 1
        if sum_el == sum(cover_arr[el][0]):
            for i in range(len(w)):
                if cover_arr[el][0][i] == 1:
                    w[i] -= 1
            del cover_arr[el]
        else:
            el += 1
    for i in cover_arr:
        new_child[i[1]] = 1
    child[:] = new_child

InputDataSet.columns= ["i", "j"]
df = InputDataSet.groupby("i").j.apply(frozenset)
n = df.shape[0]  # элементы(столбцы)
m = len(set(InputDataSet.j)) # подмножества(строки)
arr = np.zeros((m, n), dtype="object")
for i in range(n):
	for j in list(df)[i]:
        arr[j-1][i] = 1
population = populationCreator(arr, POPULATION_SIZE, n, m)

generationCounter = 0
generationToEnd = 0
bestFitness = 1000000000
bestCover = []

fitnessUpdate(population, arr)
FitnessValues = [individual.fitness.values[0] for individual in population]
print(FitnessValues)

while generationCounter < MAX_GENERATIONS or generationToEnd < MAX_GENERATIONS_BETTER:

	generationCounter += 1
	child = []
	while True:
		offspring = selTournament(population, len(population))
		offspring = list(map(clone, offspring))
		if rand.random() < P_CROSSOVER:
			child = cxOnePoint(offspring[0], offspring[1])
			if rand.random() < P_MUTATION:
				mutFlipBit(child, m)
			if checkCover(child, arr, n, m):
			     break
			#individOpt(child, arr, n, m)
			#break
	# инидвид получен и он является покрытием множества
	while True:
		index = rand.randint(0, POPULATION_SIZE - 1)
		# sub_fitness = []
		# sub_fitness.append(offspring[index])
		# if fitnessWeighted(sub_fitness, arr)[0] >= float(sum(FitnessValues)) / len(FitnessValues):
		#	offspring[index][:] = child
		#	break
		if fitnessUnweighted(offspring[index])[0] >= float(sum(FitnessValues)) / len(FitnessValues):
            offspring[index][:] = child
            break

	population[:] = offspring  ##обновление популяции
	fitnessUpdate(population, arr)
	FitnessValues = [individual.fitness.values[0] for individual in population]

	if bestFitness > min(FitnessValues):
		generationToEnd = 0
		bestFitness = min(FitnessValues)
		bestCover = population[FitnessValues.index(bestFitness)]
	else:
		generationToEnd += 1

solution = []
for i in range(len(bestCover)):
    if bestCover[i] == 1:
        solution.append(i+1)
result = pd.DataFrame(solution, columns=["RowInCover"])
OutputDataSet = result
	'
	, @input_data_1 = @requestTable
	, @params = N'@pop_size INT, @crossover FLOAT, @mutation FLOAT, @max_generation INT, @max_generation_better INT, @p FLOAT'
	, @pop_size = @pop_size
	, @crossover = @crossover
	, @mutation = @mutation
	, @max_generation = @max_generation
	, @max_generation_better = @max_generation_better
	, @p = @p

END;
