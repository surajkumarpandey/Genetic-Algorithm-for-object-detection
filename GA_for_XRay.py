import cv2,numpy,time,random
import numpy as np


img_name = 'x-ray.jpg'
img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)

color = (0,0,0)	#color of bounding box
thickness = 1	#thicknes of bounding box

h,w = img.shape

fittest = [-1,-1]
rect_coords = [0,0,0,0]

Weights = [0.1,100]
#Weights = [0.3,100]

def init_pop(pop_size):
	pop = []
	for i in range(pop_size):
		x1 = int(random.uniform(0, 1)*w)
		x2 = int(random.uniform(0, 1)*w)
		y1 = int(random.uniform(0, 1)*h)
		y2 = int(random.uniform(0, 1)*h)
		if x2>x1:		
			if y2>y1:			
				pop.append([x1,x2,y1,y2])
			else:
				pop.append([x1,x2,y2,y1])
		else:
			if y2>y1:			
				pop.append([x2,x1,y1,y2])
			else:
				pop.append([x2,x1,y2,y1])
		
	return pop

def calc_fitness(pop):
	fitness = []
	for rectangle in pop:
		filled = 0
		total_space = 0
		filled_prcntg = 0
		for i in range(rectangle[2],rectangle[3]):
			for j in range(rectangle[0],rectangle[1]):
				if img[i,j] < 120:
					filled = filled +1
				total_space += 1
		if (total_space != 0):		
			filled_prcntg =filled*100/total_space
		fitness.append([filled,filled_prcntg])
	return fitness	

def select_parents(fitness,population):
	dummy = fitness
	parents = []
	for i in range(len(population)/2):
		weighted_fitness = np.matmul(dummy,Weights)	
		max_idx = np.argmax(weighted_fitness)		#select the maximum fit member i.e. the one that has the largest filling and largest percentage in case of ties
		parents.append(population[max_idx])
		dummy[max_idx] = [-1,-1]
	return parents
	
		
					

def crossover(parents):
	l = len(parents)
	children = []
	for i in range(l):
		p1 = parents[i%len(parents)]
		p2 = parents[(i+1)%len(parents)]
		x1,x2 = random.sample([p1[0],p1[1],p2[0],p2[1]],2)
		y1,y2 = random.sample([p1[2],p1[3],p2[2],p2[3]],2)
		if x2>x1:		
			if y2>y1:			
				children.append([x1,x2,y1,y2])
			else:
				children.append([x1,x2,y2,y1])
		else:
			if y2>y1:			
				children.append([x2,x1,y1,y2])
			else:
				children.append([x2,x1,y2,y1])
	new_pop = parents + children	
	return new_pop

def mutation(population,num_mut):
	for i in range(num_mut):
		x = random.uniform(0, 1)
		mu = population[len(population)/2 -1 + int(x*(len(population)-1)/2)]	#chose the members to be mutated from the later half of the population as per elitism
		mu[0] = int(x*mu[0]) 
		mu[1] = int(x*mu[1]) 
		mu[2] = int(x*mu[2]) 
		mu[3] = int(x*mu[3])

	return population					
	
	
pop_size = 30
number_of_gens = 100
population = init_pop(pop_size)	#initialize population
num_mut = 2
f = open("Bboxuniform.txt", "w")

		
for g in range(number_of_gens):
	print("Generation: ",g)	
	img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
	
	fitness = calc_fitness(population)		#*****Fitness calculated
	weighted_fitness = np.matmul(fitness,Weights)	
	max_idx = np.argmax(weighted_fitness)		#index of element having maximum fitness and maximmum percentage in case of a tie
	if fitness[max_idx] > fittest:		
		fittest = fitness[max_idx]		#fittest member updated
		rect_coords = population[max_idx]	#coordinates of fittest member updated
	print("current fitness: ",fitness[max_idx])	
	print("fittest: ",fittest)
	fileval = fitness[max_idx][0]+fitness[max_idx][1] 
	f.write(str(fileval)+"\n") 
	
	for membrs in population:
		img = cv2.rectangle(img, (membrs[0],membrs[2]),(membrs[1],membrs[3]), color, thickness) 
		cv2.imshow('Uniform',img)
	

	img = cv2.rectangle(img, (rect_coords[0],rect_coords[2]),(rect_coords[1],rect_coords[3]), color, 4*thickness)
	cv2.imshow('Uniform',img)
	cv2.waitKey(1)

	parents = select_parents(fitness,population)	#*****Select parents for mating 
	population = crossover(parents)			#*****Perform crossover
	population = mutation(population,num_mut)	#*****Perform mutation

print("Box found!")
time.sleep(10)	
f.close() 
cv2.destroyAllWindows()
