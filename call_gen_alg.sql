use [CoverTable]


DECLARE 
@table1 VARCHAR(80), 
@pop_size INT,  
@crossover FLOAT,
@mutation FLOAT,
@max_generation INT,
@max_generation_better INT,
@p FLOAT

SET @table1 = 'Cvt';   -- название таблицы источника таблицы, формат: Row INT, Col INT
SET @pop_size = 50;
SET	@crossover = 1.0;
SET @mutation = 1.0;
SET	@max_generation = 1000;
SET @max_generation_better = 500;
SET	@p = 0.5;

EXECUTE Genetic_Algorithm @table1, @pop_size, @crossover, @mutation, @max_generation, @max_generation_better, @p;

SELECT * FROM Genetic_Algorithm_Result;
