import pygame
from pygame.locals import *
import random
import numpy
import math
import pygad
import pygad.nn
import pygad.gann
import math
import matplotlib.pyplot as plt

game_close = False

#DOe de diagonal input dude niet lui zijn ik weet dat het ez is gwn u kop gebruiken regel 329

pygame.init()

pi = 3.1415

x_width = 750
y_width = 750
refdist = 41.0122

displayRef = pygame.display.set_mode((x_width, y_width))
pygame.display.set_caption("snAIke")

blue = (0, 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
black = (0, 0, 0)

poslog = []
dislog = [] 
anglelog = []
dirlog = []
eatlog = []
evallog = []

fastsearch = True
#fitness value

fitness = 0

displaybool = False


font_style = pygame.font.SysFont(None, 40)

displayRef.fill(black)

def WeightsCalc(inputl, hiddenl, outputl):
    weights = inputl*hiddenl[0]+hiddenl[-1]*outputl
    for i in range(len(hiddenl)-1):
        weights += hiddenl[i]*hiddenl[i+1]
    print("number of weights : ", weights, "\n")
    return weights

def FoodCheck(foodx, foody, snakex, snakey):

    ret = False
    
    for i in range(len(snakex)):
        if snakex[i] == foodx:
            if snakey[i] == foody:
                ret = True
    return ret

def Sort(sub_li):
    
    sub_li.sort(key = lambda x: x[0], reverse=True)
    return sub_li

def on_parents(ga_instance, selected_parents):
    global fastsearch
    global number_weights
    global num_parents_mating
    global fitnessarr
    global GANN_instance
    global num_generations
    global fitarrcopy
    k =10 #mutation precision
    initrange = 4
    fitarrcopy = Sort(fitnessarr)
    print("GEN High Score : ", fitarrcopy[0][0]) 
    fitarrcopy = fitarrcopy[: num_parents_mating]
    for i in range(0, num_parents_mating):
        selected_parents = numpy.delete(selected_parents, i, 0)
        selected_parents = numpy.insert(selected_parents, i, ga_instance.population[fitnessarr.index(fitarrcopy[i])], 0)
    fitnessarr = []
    for i in range(0, len(ga_instance.population)):

                
        fitindex = i%len(selected_parents)
        arr = selected_parents[fitindex].copy()

        if i < len(ga_instance.population)-num_parents_mating:
            
            mutation_range = 0.2*numpy.exp(-0.000025*fitarrcopy[fitindex][0])
##            mutation_range = 0.2
            
            for j in range(len(arr)):
                if random.random() <= 1/number_weights:    
                    addmin = numpy.random.choice([-1,1])
                    mutval = 2**(-random.uniform(0,k))
                    arr[j] += mutation_range*initrange*addmin*mutval
                
            for j in range(0, len(arr)):
                if arr[j] > 4:
                    arr[j] = 4
                elif arr[j] < -4:
                    arr[j] = -4
                   
        arr = numpy.reshape(arr, (1,number_weights))

        ga_instance.population = numpy.delete(ga_instance.population, i, 0)
        ga_instance.population = numpy.insert(ga_instance.population, i, arr, axis=0)
    
def predict(inputarray, sol_idx):
        
    return pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=inputarray)

def fitness_func(s2lution, sol_idx):
    global idx
    global fitnessarr
    global GANN_instance
    if sol_idx ==0:
        print("NEW GEN")
        
    fitness = gameloop(sol_idx, True, 0)
    fitnessarr.append([fitness, sol_idx])
    return fitness

def callback_generation(ga_instance):
    
    global GANN_instance
    global num_parents_mating
    global fitarrcopy
    
    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)
    print("END GEN {generation} \n".format(generation=ga_instance.generations_completed))

#logging function
def protoEval(dislog, poslog, score, looped, sol_idx):

    fitness = 100*score

    if score >= 100:
        print("score: ", score, "sol index:", sol_idx)
    if score <= 5:
        for i in range(0, len(dislog)-1):
            if (dislog[i] - dislog[i+1]) > 0 and i not in eatlog:
                fitness+=1
            elif (dislog[i] - dislog[i+1]) < 0 and i not in eatlog:
                fitness-=1
    return fitness



def snakeLog(xpos, ypos, distance):
    poslog.append([xpos, ypos])
    dislog.append(distance)

#text function
def message(msg, color, xoffset, yoffset):
    renderedmsg = font_style.render(msg, True, color)
    displayRef.blit(renderedmsg, [x_width // 2 - xoffset, y_width // 2 - yoffset])

#hit checker
def hit(xbody, ybody):
    hitVal = False
    for i in range(1, len(xbody)):
        if xbody[0] == xbody[i] and ybody[0] == ybody[i]:
            hitVal = True
    return hitVal

#main game loop
def gameloop(sol_idx, show, delay):

   
    
  
    global displaybool
    
    #declaring variables
    score = 0
    iterations = 0
    move = 0
    fitness = 0

    global poslog
    global dislog
    global anglelog
    global dirlog
    global eatlog
    global evallog
    
    
    poslog = []
    dislog = []
    anglelog = []
    dirlog = []
    eatlog = []
    evallog = []

    x_change = 25
    y_change = 0

    running = True
    game_close = False
    foodThere = False
    buttonPushed = False
    looped = False


    foodx = 0
    foody = 0
    
    #networkinput
    x = 1
    Y = 0

    bodyWarn = 0
    edgeWarn = 0
    leftSoftWarn = False
    rightSoftWarn = False
    topSoftWarn = False
    bottomSoftWarn = False

    snakeBodyx = [x_width // 2, x_width // 2 - 25, x_width // 2 - 50]
    snakeBodyy = [y_width // 2, y_width // 2, y_width // 2]

    if not displaybool:
        displaybool = True
        if not show:
            message("DISPLAY DISABLED", white, 350,0)
            pygame.display.update()
            
    while running:
        while game_close:
            if show:
                displayRef.fill(black)
                pygame.display.update()
            return protoEval(dislog, poslog, score, looped, sol_idx)

        if not foodThere:
            foodx = random.randint(0, 749)
            foodx -= foodx % 25
            while foodx in snakeBodyx:
                foodx = random.randint(0, 749)
                foodx -= foodx % 25
            foody = random.randint(0, 749)
            foody -= foody % 25

            while foody in snakeBodyy:
                foody = random.randint(0, 749)
                foody -= foody % 25
            foodThere = True
            
        #network input gathering
        foodOnLeft = 0
        foodOnRight = 0
        distance = math.sqrt(int(((snakeBodyx[0]-foodx)/25)**2)+(((snakeBodyy[0]-foody)/25)**2))
        if snakeBodyy[1]-snakeBodyy[0] == -25:
            y = -1
            x = 0
        elif snakeBodyy[1]-snakeBodyy[0] == 25:
            y = 1
            x = 0
        elif snakeBodyx[1]-snakeBodyx[0] == 25:
            y = 0
            x = -1
        else:
            y = 0
            x = 1
    
        leftSoftWarn = (30-snakeBodyx[0]/25)/30
        rightSoftWarn = (1+snakeBodyx[0]/25)/30
        topSoftWarn = (30-snakeBodyy[0]/25)/30
        bottomSoftWarn = (1+snakeBodyy[0]/25)/30

        leftUpSoftWarn = 0
        leftDownSoftWarn = 0
        rightUpSoftWarn = 0
        rightDownSoftWarn = 0

        PleftUpSoftWarn = 0
        PleftDownSoftWarn = 0
        PrightUpSoftWarn = 0
        PrightDownSoftWarn = 0
        
        perspRight = 0
        perspLeft = 0
        perspTop = 0
        

        
        angle = math.atan(numpy.abs(foody-snakeBodyy[0])/numpy.abs(foodx-snakeBodyx[0]))
        
        if math.isnan(angle):
            angle = 0
        angle = math.trunc(angle*10000)/10000
       
        reserveangle = angle
        
        for i in range(1, len(snakeBodyx)):
            if snakeBodyx[0] != snakeBodyx[i] and snakeBodyy[0] == snakeBodyy[i]:
                if snakeBodyx[0] < snakeBodyx[i]:
                    if (30-(snakeBodyx[i] - snakeBodyx[0])/25)/29 > rightSoftWarn:
                        rightSoftWarn = (30-(snakeBodyx[i] - snakeBodyx[0])/25)/29
                else:
                    if (30-(snakeBodyx[0] - snakeBodyx[i])/25)/29 > leftSoftWarn:
                        leftSoftWarn = (30-(snakeBodyx[0] - snakeBodyx[i])/25)/29
                        
        for i in range(1, len(snakeBodyx)):
            if snakeBodyx[0] == snakeBodyx[i] and snakeBodyy[0] != snakeBodyy[i]:
                if snakeBodyy[0] < snakeBodyy[i]:
                    if (30 - (snakeBodyy[i]-snakeBodyy[0])/25)/29 > bottomSoftWarn:
                            bottomSoftWarn = (30 - (snakeBodyy[i]-snakeBodyy[0])/25)/29 
                else:
                    if (30 - (snakeBodyy[0]-snakeBodyy[i])/25)/29 > topSoftWarn:
                            topSoftWarn = (30 - (snakeBodyy[0]-snakeBodyy[i])/25)/29
##loop telkens een blok verder totdat je iets raakt stop een per een dan stop loop (seperate kan ook kijk wat gemakkelijker plz niet te veel booleans das niet clean
        diagonalInputLoop = True
        for diagonalInputLoop:
            if
            
        #direction based input
        if x == 1:
            perspRight = bottomSoftWarn
            perspLeft = topSoftWarn
            perspTop = rightSoftWarn
            if foodx <= snakeBodyx[0]:
                angle = pi - angle
            if foody > snakeBodyy[0]:
                foodOnRight = 1
            elif foody < snakeBodyy[0]:
                foodOnLeft = 1
                
                
        elif x == -1:
            perspRight = topSoftWarn
            perspLeft = bottomSoftWarn
            perspTop = leftSoftWarn
            if foodx >= snakeBodyx[0]:
                angle = pi - angle
            if foody > snakeBodyy[0]:
                foodOnLeft = 1
            elif foody < snakeBodyy[0]:
                foodOnRight = 1
                
        elif y == 1:
            perspRight = rightSoftWarn
            perspLeft = leftSoftWarn
            perspTop = topSoftWarn
            
            if foody >= snakeBodyy[0]:
                angle += math.trunc(pi/2*10000)/10000
            else:
                angle = numpy.abs(angle - math.trunc(pi/2*10000)/10000)
            if foodx > snakeBodyx[0]:
                foodOnRight = 1
            elif foodx < snakeBodyx[0]:
                foodOnLeft = 1
                
            

        elif y == -1:
            perspRight = leftSoftWarn
            perspLeft = rightSoftWarn
            perspTop = bottomSoftWarn
            
            if foody <= snakeBodyy[0]:
                angle += math.trunc(pi/2*10000)/10000
                
            else:
                angle = numpy.abs(angle - math.trunc(pi/2*10000)/10000)
                
            if foodx > snakeBodyx[0]:
                foodOnLeft = 1
                
            elif foodx < snakeBodyx[0]:
                foodOnRight = 1

        #normalizing angle and distance
        angle = math.trunc(angle/pi*10000)/10000
        distance = math.trunc(distance/41.012*1000)/1000
        
        #logging
        snakeLog(snakeBodyx[0], snakeBodyy[0], distance)
        if perspLeft < 0.99:
            perspleft = 0
        if perspRight < 0.99:
            perspRight = 0
        if perspTop < 0.99:
            perspTop = 0
##        perspLeft = perspLeft**6
##        perspRight = perspRight**6
##        perspTop = perspTop**6
        inputPassThroughArr = numpy.array([[angle, foodOnLeft, foodOnRight, perspLeft, perspTop, perspRight]])

##  the following inputs are being used:
##    - perspective (danger) from top 1 block
##    - perspective (danger) from left 1 block
##    - perspective (danger) from right 1 block
##    - closest angle between top head and food
##    - indication of the food being left or right (two seperate inputs)
##    - distance to food
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                
        move = predict(inputPassThroughArr, sol_idx)

        if x == 1:
            if move[0] == 0:
                x_change = 25
                y_change = 0

            elif move[0] == 1:
                x_change = 0 
                y_change = -25

            elif move[0] == 2:
                x_change = 0
                y_change = 25
            
        elif x == -1:
            
            if move[0] == 0:
                x_change = -25 
                y_change = 0

            elif move[0] == 1:
                x_change = 0
                y_change = 25

            elif move[0] == 2:
                x_change = 0
                y_change = -25
            
        elif y == 1:
            
            if move[0] == 0:
                x_change = 0
                y_change = -25

            elif move[0] == 1:
                x_change = -25
                y_change = 0

            elif move[0] == 2:
                x_change = 25
                y_change = 0
            
        elif y == -1:
            
            if move[0] == 0:
                x_change = 0
                y_change = 25

            elif move[0] == 1:
                x_change = 25
                y_change = 0

            elif move[0] == 2:
                x_change = -25
                y_change = 0
                              
        for i in range(len(snakeBodyx) - 1, 0, -1):
            snakeBodyx[i] = snakeBodyx[i - 1]
            snakeBodyy[i] = snakeBodyy[i - 1]

        snakeBodyx[0] += x_change
        snakeBodyy[0] += y_change
        
        if snakeBodyx[0] == foodx and snakeBodyy[0] == foody:
            score += 1
            lastBlockX = snakeBodyx[len(snakeBodyx) - 1]
            lastBlockDiffX = snakeBodyx[len(snakeBodyx) - 1] - snakeBodyx[len(snakeBodyx) - 2]
            lastBlockY = snakeBodyy[len(snakeBodyy) - 1]
            lastBlockDiffY = snakeBodyy[len(snakeBodyy) - 1] - snakeBodyy[len(snakeBodyy) - 2]
            snakeBodyx.append(lastBlockX - lastBlockDiffX)
            snakeBodyy.append(lastBlockY - lastBlockDiffY)
            foodThere = False
            foodx = random.randint(0, 749)
            foodx -= foodx % 25
            foody = random.randint(0, 749)
            foody -= foody % 25
            eatlog.append(iterations)
            evallog.clear()
            
            while FoodCheck(foodx, foody, snakeBodyx, snakeBodyy):
                foodx = random.randint(0, 749)
                foodx -= foodx % 25
                foody = random.randint(0, 749)
                foody -= foody % 25
            foodThere = True
            
        evallog.append([snakeBodyx[0], snakeBodyy[0]])
        
        if snakeBodyx[0] >= x_width or snakeBodyx[0] < 0 or snakeBodyy[0] >= y_width or snakeBodyy[0] < 0 or hit(
                snakeBodyx, snakeBodyy):
            game_close = True


        if evallog.count([snakeBodyx[0], snakeBodyy[0]]) > 3:
            game_close = True
            looped = True
            
        # surfaceFill
        displayRef.fill(black)
        if show:
            
            if foodThere:
                pygame.draw.rect(displayRef, red, [foodx, foody, 25, 25])
            for i in range(len(snakeBodyx)):
                if i == 0:
                    pygame.draw.rect(displayRef, green, [snakeBodyx[i], snakeBodyy[i], 25, 25])
                else:
                    pygame.draw.rect(displayRef, blue, [snakeBodyx[i], snakeBodyy[i], 25, 25])
            message(str(score), white, 320, -100)
            message(str(sol_idx), white, 320, 10)
            pygame.display.update()
            buttonPushed = False
            if score>=1:
                pygame.time.wait(delay)
        iterations += 1
        
print("RANDOM LOCAL SEARCH INITIALISATION\n")


checklist = []    
fitnessarr = []
fitarrcopy = []

num_classes = 3
num_neurons_hidden_layers=[10,  10]
num_neurons_input=6

num_solutions = 100

GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_neurons_input,
                                num_neurons_hidden_layers=num_neurons_hidden_layers,
                                num_neurons_output=num_classes,
                                hidden_activations=["relu", "relu"],
                                output_activation="softmax")

print("input layer : ", num_neurons_input)
print("hidden layers : ",num_neurons_hidden_layers)
print("output layer : ", num_classes)


number_weights = WeightsCalc(num_neurons_input, num_neurons_hidden_layers,num_classes)
population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)


initial_population = population_vectors.copy()

num_parents_mating = 5

num_generations = 100
parent_selection_type = "rank" 

mutation_type = None


keep_parents = 0

print("population = ", num_solutions)
print("mating selection = ", num_parents_mating)


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=None,
                       init_range_low=-4,
                       on_parents=on_parents,
                       init_range_high=4,
                       mutation_type=mutation_type,
                       keep_parents=keep_parents,
                       on_generation=callback_generation,
                       suppress_warnings=True)
    
ga_instance.run()
ga_instance.plot_result()

