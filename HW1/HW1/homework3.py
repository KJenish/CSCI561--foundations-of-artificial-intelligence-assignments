import numpy as np
import random
import pandas as pd
import operator

class city:
    def __init__(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z
    
    def distance(self,city):
        xdistance = abs(self.x - city.x)
        ydistance = abs(self.y - city.y)
        zdistance = abs(self.z - city.z)
        total_distance = np.sqrt((xdistance ** 2) + (ydistance ** 2)+(zdistance ** 2))
        #total_distance = distance.euclidean(xdistance,xdistance,xdistance)
        return total_distance
    
    def __repr__(self):
        return str(self.x) +" "+ str(self.y)+" " + str(self.z)
    
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.00
    
    def routeDistance(self):
        if self.distance ==0:
            routeDistance = 0
            for i in range(len(self.route)):
                startCity = self.route[i]
                nextCity = None
                if (i + 1) < len(self.route):
                    nextCity = self.route[i + 1]
                else:
                    nextCity = self.route[0]
                routeDistance = routeDistance + startCity.distance(nextCity)
            self.distance = routeDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
    
def makeRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def firstPopulation(populationSize, cityList):
    population = []

    if(len(cityList)<=50):
        populationSize=100
        for i in range(populationSize):
            population.append(makeRoute(cityList))
    else:
        for i in range(populationSize):
            population.append(makeRoute(cityList))
    return population

def fitnessRoutes(population):
    fitnessScores = {}
    for i in range(0,len(population)):
        fitnessScores[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessScores.items(), key = operator.itemgetter(1), reverse = True)

def selectionRes(populationRanked, eliteSize):
    selectionResults = []
    
    dataframe = pd.DataFrame(np.array(populationRanked), columns=["Index","Fitness"])
    dataframe['cum_sum'] = dataframe.Fitness.cumsum()
    dataframe['cum_perc'] = round(100*dataframe.cum_sum/dataframe.Fitness.sum())
    
    for i in range(eliteSize):
        selectionResults.append(populationRanked[i][0])
    for i in range(len(populationRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(len(populationRanked)):
            if pick <= dataframe.iat[i,3]:
                selectionResults.append(populationRanked[i][0])
                break
    return selectionResults

def parentPool(population, selectionResults):
    parentpool = []
    for i in range(len(selectionResults)):
        index = selectionResults[i]
        parentpool.append(population[index])
    return parentpool

def crossover(parent1, parent2):
    child = []
    cP1 = []
    cP2 = []
    
    generationA = int(random.random() * len(parent1))
    generationB = int(random.random() * len(parent1))
    
    startGene = min(generationA, generationB)
    endGene = max(generationA, generationB)

    for i in range(startGene, endGene):
        cP1.append(parent1[i])
        
    cP2 = [item for item in parent2 if item not in cP1]

    child = cP1 + cP2
    return child

def generatePopulation(mpool, eSize):
    childlist = []
    tlength = len(mpool) - eSize
    pool = random.sample(mpool, len(mpool))

    for i in range(eSize):
        childlist.append(mpool[i])
    
    for i in range(tlength):
        child = crossover(pool[i], pool[len(mpool)-i-1])
        childlist.append(child)
    return childlist

def mutate(individual, mutationRate):
    for x in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[x]
            city2 = individual[swapWith]
            
            individual[x] = city2
            individual[swapWith] = city1
    return individual

def mutatePop(population, mutationRate):
    mutatedPopulation = []
    
    for index in range(len(population)):
        mutatedIndex = mutate(population[index], mutationRate)
        mutatedPopulation.append(mutatedIndex)
    return mutatedPopulation

def newGeneration(firstGen, eliteSize, mutationRate):
    populationRanked = fitnessRoutes(firstGen)
    selectionResults = selectionRes(populationRanked, eliteSize)
    parentpool = parentPool(firstGen, selectionResults)
    children = generatePopulation(parentpool, eliteSize)
    nextGeneration = mutatePop(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, populationSize, eliteSize, mutationRate, generations):
    pop = firstPopulation(populationSize, population)
        
    if(len(population)<=50):
        generations=400
        for i in range(generations):
            pop = newGeneration(pop, eliteSize, mutationRate)
    elif (100 < len(population)<=200):
        generations=550
        for i in range(generations):
            pop = newGeneration(pop, eliteSize, mutationRate)
    elif(len(population)>200):
        generations=700
        for i in range(generations):
            pop = newGeneration(pop, eliteSize, mutationRate)
    else:
        for i in range(generations):
            pop = newGeneration(pop, eliteSize, mutationRate)
    bestRouteIndex = fitnessRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    bestRoute.append(bestRoute[0])
    return bestRoute

cityList = []
#with open('../resource/asnlib/public/sample/input3.txt') as f:
with open('input.txt') as f:
    next(f)
    for line in f:
            x1,y1,z1 = line.split()
            cityList.append(city(x=int(x1),y=int(y1),z=int(z1)))
            
ans=geneticAlgorithm(population=cityList, populationSize=110, eliteSize=20, mutationRate=0.01, generations=500)
#print(ans)
f = open("output.txt", "w")
for city in ans:
    f.write(str(city.x) + " ")
    f.write(str(city.y) + " ")
    f.write(str(city.z) + " ")
    f.writelines("\n")
    #print(cities[city])
f.close()