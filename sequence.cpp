#include <iostream>
#include <ctime>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <algorithm>

using namespace std;

typedef struct Individual {
    int* chromosome;
    int fitness;
} Individual;

int calculateFitness(int** adj, int* chromosome, int V) {
    int fitness = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = v; u < V; ++u) {
            if (adj[v][u] && chromosome[v] == chromosome[u]) {
                ++fitness;
            }
        }
    }
    return fitness;
}

int getRandomNumber(int min, int max) {
    return min + rand() % ((max + 1) - min);
}

void initializePopulation(Individual* population, int populationSize, int V, int numColors) {
    for (int i = 0; i < populationSize; ++i) {
        population[i].chromosome = new int[V];
        for (int j = 0; j < V; ++j) {
            population[i].chromosome[j] = getRandomNumber(0, numColors - 1);
        }
    }
}

Individual tournamentSelection(const Individual* population, int populationSize) {
    int tournamentSize = 3;
    Individual bestIndividual = population[getRandomNumber(0, populationSize - 1)];
    for (int i = 1; i < tournamentSize; ++i) {
        Individual candidate = population[getRandomNumber(0, populationSize - 1)];
        if (candidate.fitness < bestIndividual.fitness) {
            bestIndividual = candidate;
        }
    }
    return bestIndividual;
}

void crossover(const Individual& parent1, const Individual& parent2, Individual& offspring1, Individual& offspring2, int V) {
    int crossoverPoint = getRandomNumber(0, V - 1);
    for (int i = 0; i < V; ++i) {
        if (i <= crossoverPoint) {
            offspring1.chromosome[i] = parent1.chromosome[i];
            offspring2.chromosome[i] = parent2.chromosome[i];
        }
        else {
            offspring1.chromosome[i] = parent2.chromosome[i];
            offspring2.chromosome[i] = parent1.chromosome[i];
        }
    }
}

void mutate1(Individual& individual, int numColors, int V) {
    int mutationPoint = getRandomNumber(0, V - 1);
    individual.chromosome[mutationPoint] = getRandomNumber(0, numColors - 1);
}

void swapMutation(Individual& individual, int V) {
    int point1 = getRandomNumber(0, V - 1);
    int point2 = getRandomNumber(0, V - 1);
    while (point1 == point2) {
        point2 = getRandomNumber(0, V - 1);
    }
    swap(individual.chromosome[point1], individual.chromosome[point2]);
}

void shiftMutation(Individual& individual, int numColors, int V) {
    int start = getRandomNumber(0, V - 1);
    int end = getRandomNumber(0, V - 1);
    while (start == end) {
        end = getRandomNumber(0, V - 1);
    }
    if (start > end) {
        swap(start, end);
    }
    for (int i = start; i <= end; ++i) {
        individual.chromosome[i] = (individual.chromosome[i] + 1) % numColors;
    }
}

void recombinationMutation(Individual& individual1, Individual& individual2, int V) {
    for (int i = 0; i < V; ++i) {
        if (rand() % 2 == 0) {
            swap(individual1.chromosome[i], individual2.chromosome[i]);
        }
    }
}

void mutate(Individual& individual, Individual& individual2, int numColors, int V) {
    int mutationType = getRandomNumber(0, 3);
    switch (mutationType) {
    case 0:
        swapMutation(individual, V);
        break;
    case 1:
        shiftMutation(individual, numColors, V);
        break;
    case 2:
        recombinationMutation(individual, individual2, V);
        break;
    case 3:
        mutate1(individual, numColors, V);
        break;
    }
}

bool compareFitness(const Individual& a, const Individual& b) {
    return a.fitness < b.fitness;
}

void sortPopulation(Individual* population, int populationSize) {
    sort(population, population + populationSize, compareFitness);
}

Individual geneticAlgorithm(int** adj, int V, int populationSize, int numGenerations, int numColors) {
    Individual* population = new Individual[populationSize];

    initializePopulation(population, populationSize, V, numColors);

    for (int i = 0; i < populationSize; i++) {
        population[i].fitness = calculateFitness(adj, population[i].chromosome, V);
    }

    sortPopulation(population, populationSize);

    Individual bestIndividual = population[0];

    for (int generation = 0; generation < numGenerations; ++generation) {
        Individual* newPopulation = new Individual[populationSize];
        initializePopulation(newPopulation, populationSize, V, numColors);

        for (int i = 0; i < populationSize / 2; i++) {
            Individual parent1 = tournamentSelection(population, populationSize);
            Individual parent2 = tournamentSelection(population, populationSize);
            Individual offspring1;
            offspring1.chromosome = new int[V];
            Individual offspring2;
            offspring2.chromosome = new int[V];
            crossover(parent1, parent2, offspring1, offspring2, V);

            if (getRandomNumber(0, 100) < 60) {
                mutate(offspring1, offspring2, numColors, V);
            }

            offspring1.fitness = calculateFitness(adj, offspring1.chromosome, V);
            offspring2.fitness = calculateFitness(adj, offspring2.chromosome, V);

            newPopulation[i] = offspring1;
            newPopulation[i + populationSize / 2] = offspring2;
        }

        for (int i = 0; i < populationSize; ++i) {
            delete[] population[i].chromosome;
            population[i] = newPopulation[i];
        }
        delete[] newPopulation;
        sortPopulation(population, populationSize);

        if (population[0].fitness < bestIndividual.fitness) {
            bestIndividual = population[0];
        }

        //cout << "Generation " << generation << " Best Fitness: " << bestIndividual.fitness << endl;

        if (bestIndividual.fitness == 0) break;
    }

    for (int i = 0; i < populationSize; ++i) {
        if (population[i].chromosome != bestIndividual.chromosome) {
            delete[] population[i].chromosome;
        }
    }
    delete[] population;
    return bestIndividual;
}

int readMatrix(int size, int** a, const char* filename) {
    FILE* pf = fopen(filename, "r");
    if (pf == NULL) {
        cerr << "Błąd otwarcia pliku" << endl;
        return 0;
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (fscanf(pf, "%d", &a[i][j]) != 1) {
                cerr << "Błąd odczytu elementu macierzy na pozycji (" << i << ", " << j << ")" << endl;
                fclose(pf);
                return 0;
            }
        }
    }

    fclose(pf);
    return 1;
}

int main() {
    int populationSize = 100;
    int numGenerations = 3000;
    int numColors = 15;
    int N = 450;
    int** adj = new int* [N];
    if (adj == NULL) {
        printf("Memory allocation error for adj");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        adj[i] = new int[N];
        if (adj[i] == NULL) {
            printf("Memory allocation error for adj %d\n", i);
            return 1;
        }
    }

    const char* filename = "adjacency_matrix.txt";
    if (!readMatrix(N, adj, filename)) {
        cerr << "Error reading matrix from file" << endl;
        return 1;
    }

    srand(time(0));

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    Individual bestSolution = geneticAlgorithm(adj, N, populationSize, numGenerations, numColors);
    end = clock();

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    cout << "Time: " << cpu_time_used << " seconds" << endl;
    cout << "Best Solution Fitness: " << bestSolution.fitness << endl;
    cout << "Coloring: ";
    for (int i = 0; i < N; ++i) {
        cout << bestSolution.chromosome[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < N; ++i) {
        delete[] adj[i];
    }
    delete[] adj;
    delete[] bestSolution.chromosome;

    return 0;
}
