import cv2
from time import sleep
import numpy as np
import queue
import random
import pickle as pkl

exitColor = [0,0,255]
baseColor = [0,0,0]
backgroundColor = [255,255,255]

draw = True
cv2.namedWindow('display')

################################################################################
# Agent stuff
################################################################################

personList = []
r = 3 # resolution in pixels/meter
spawnRate = 0.001

class Person:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.closePeople = []
        self.speed = (1 + 2*random.random())
        self.exited = False
        self.radius = (0.25 + 0.5*random.random())

def letPeopleMove():

    for person in personList:
        if person.exited:
            continue
        x = int(person.x)
        y = int(person.y)

        shortestDir = dirsArray[exit1Dirs[y][x]]
        avoidWallsDir = dirsArray[wallDirs[y][x]]
        wallDistance = wallDist[y][x]
        wallAvoidanceFactor = -2/(wallDistance+0.2)

        numCollisions = 0
        desiredX = 0
        desiredY = 0

        for j in person.closePeople:
            otherPerson = personList[j]
            xdir = (person.x-otherPerson.x)/r
            ydir = (person.y-otherPerson.y)/r
            dist = ((xdir)**2 + (ydir)**2)**(.5)
            desiredX += xdir / (dist+1)
            desiredY += ydir / (dist+1)
            if dist < person.radius + otherPerson.radius + 1:
                numCollisions += 1
        
        speedFactor = 1 / (numCollisions + 1)

        desiredX += person.speed * speedFactor * shortestDir[1] \
                            + wallAvoidanceFactor * avoidWallsDir[1]
        desiredY += person.speed * speedFactor * shortestDir[0] \
                            + wallAvoidanceFactor * avoidWallsDir[0]

        
        desiredX = person.x + r*desiredX/3
        desiredY = person.y + r*desiredY/3
        newLoc = exit1Dist[int(desiredY)][int(desiredX)]
        if newLoc < 100000:
            person.x = desiredX
            person.y = desiredY

        if exit1Dist[y][x] < 2:
            person.exited = True

def spawnPeople():

    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if not (map[y][x]==backgroundColor).all():
                if random.random() < spawnRate:
                    p = Person(x,y)
                    personList.append(p)
                
def determineClosePeople():
    b = 20 # bin size
    distThresh = 15 # distance needed to be "close"
    xx = int(map.shape[1] / b) + 2
    yy = int(map.shape[0] / b) + 2
    bins = [[[] for j in range(xx)] for i in range(yy)]
    for i,p1 in enumerate(personList):
        if p1.exited:
            continue
        p1.closePeople = []
        bins[int(p1.y/b)][int(p1.x/b)].append(i)
    
    for y in range(yy-1):
        for x in range(xx-1):
            # bin vs. itself
            for i in bins[y][x]:
                p1 = personList[i]
                for j in bins[y][x]:
                    if i>=j: continue
                    p2 = personList[j]
                    if ((p1.x-p2.x) ** 2 + (p1.y-p2.y) ** 2)**(.5) < distThresh*r:
                        p1.closePeople.append(j)
                        p2.closePeople.append(i)
            # bin vs neighbors
            for i in bins[y][x]:
                p1 = personList[i]
                for j in bins[y+1][x]:
                    p2 = personList[j]
                    if ((p1.x-p2.x) ** 2 + (p1.y-p2.y) ** 2)**(.5) < distThresh*r:
                        p1.closePeople.append(j)
                        p2.closePeople.append(i)
            for i in bins[y][x]:
                p1 = personList[i]
                for j in bins[y+1][x+1]:
                    p2 = personList[j]
                    if ((p1.x-p2.x) ** 2 + (p1.y-p2.y) ** 2)**(.5) < distThresh*r:
                        p1.closePeople.append(j)
                        p2.closePeople.append(i)
            for i in bins[y][x]:
                p1 = personList[i]
                for j in bins[y][x+1]:
                    p2 = personList[j]
                    if ((p1.x-p2.x) ** 2 + (p1.y-p2.y) ** 2)**(.5) < distThresh*r:
                        p1.closePeople.append(j)
                        p2.closePeople.append(i)

################################################################################
# UTIL
################################################################################

def loadMap(mapName):
    img = cv2.imread("maps/" + mapName + ".png", cv2.IMREAD_COLOR)
    return img

dirsArray = [(-1,0),(1,0),(0,1),(0,-1),(0,0),(-0.71,-0.71),(0.71,-0.71),(0.71,0.71),(-0.71,0.71)]

def computeDistanceToWalls(map, wallColor):
    grid = np.full((map.shape[0], map.shape[1]), 100000.0)
    dirs = np.full((map.shape[0], map.shape[1]), 4, np.uint8)
    visited = np.full((map.shape[0], map.shape[1]), 0)
    gridimg = np.full((map.shape[0], map.shape[1], 3), 255, np.uint8)
    q = queue.Queue()

    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if (map[y][x]==backgroundColor).all():
                grid[y][x] = 0
                gridimg[y][x][1] = 0
                gridimg[y][x][0] = 0
                q.put((y,x))

    i = 0
    while q.qsize() > 0:

        y,x = q.get()
        if visited[y][x] > 0:
            continue
        if y < 1 or x < 1 or y > map.shape[0]-2 or x > map.shape[1]-2:
            continue
        visited[y][x] = 1

        a = [grid[y-1][x]+1, grid[y+1][x]+1, grid[y][x+1]+1, 
                    grid[y][x-1]+1, grid[y][x], grid[y-1][x-1]+1.5,
                    grid[y+1][x-1]+1.5, grid[y+1][x+1]+1.5, grid[y-1][x+1]+1.5]
        dirs[y][x] = np.argmin(a)
        grid[y][x] = min(a)
        q.put((y-1,x))
        q.put((y+1,x))
        q.put((y,x+1))
        q.put((y,x-1))
        gridimg[y][x][2] = min(255,int(grid[y][x]*3))

    #cv2.imshow('display', gridimg)
    #cv2.waitKey(0)
    return grid, dirs

def computeShortestPaths(map, exitColor):
    grid = np.full((map.shape[0], map.shape[1]), 100000.0)
    dirs = np.full((map.shape[0], map.shape[1]), 4, np.uint8)
    gridimg = np.full((map.shape[0], map.shape[1], 3), 255, np.uint8)
    visited = np.full((map.shape[0], map.shape[1]), 0)
    q = queue.Queue()

    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if (map[y][x]==exitColor).all():
                gridimg[y][x][1] = 0
                gridimg[y][x][0] = 0
                grid[y][x] = 0
                q.put((y,x))

    i = 0
    while q.qsize() > 0:

        y,x = q.get()
        if visited[y][x] > 0:
            continue
        if y < 1 or x < 1 or y > map.shape[0]-2 or x > map.shape[1]-2:
            continue
        if (map[y][x]==backgroundColor).all():
            continue
        visited[y][x] = 1

        a = [grid[y-1][x]+1, grid[y+1][x]+1, grid[y][x+1]+1, 
                    grid[y][x-1]+1, grid[y][x], grid[y-1][x-1]+1.414,
                    grid[y+1][x-1]+1.414, grid[y+1][x+1]+1.414, grid[y-1][x+1]+1.414]
        dirs[y][x] = np.argmin(a)
        wallPenalty = max(0, 10-wallDist[y][x])
        grid[y][x] = min(a) + 2*wallPenalty
        
        q.put((y-1,x))
        q.put((y+1,x))
        q.put((y,x+1))
        q.put((y,x-1))
        gridimg[y][x][2] = min(255,int(grid[y][x]/1.5))

    #cv2.imshow('display', gridimg)
    #cv2.waitKey(0)
    return grid, dirs

################################################################################
# SIMULATION
################################################################################


mapName = 'testmap'
map = loadMap(mapName)

wallDist, wallDirs, exit1Dist, exit1Dirs = (None,None,None,None)

try:
    wallDist = pkl.load(open('temp/' + mapName + 'wallDist.pkl', 'rb'))
    wallDirs = pkl.load(open('temp/' + mapName + 'wallDirs.pkl', 'rb'))
    exit1Dist = pkl.load(open('temp/' + mapName + 'exit1Dist.pkl', 'rb'))
    exit1Dirs = pkl.load(open('temp/' + mapName + 'exit1Dirs.pkl', 'rb'))
except Exception as e:
    print('recomputing distances...')
    wallDist, wallDirs = computeDistanceToWalls(map, backgroundColor)
    exit1Dist, exit1Dirs = computeShortestPaths(map, exitColor)
    pkl.dump(wallDist,open('temp/' + mapName + 'wallDist.pkl', 'wb'))
    pkl.dump(wallDirs,open('temp/' + mapName + 'wallDirs.pkl', 'wb'))
    pkl.dump(exit1Dist,open('temp/' + mapName + 'exit1Dist.pkl', 'wb'))
    pkl.dump(exit1Dirs,open('temp/' + mapName + 'exit1Dirs.pkl', 'wb'))
    print('...done')



spawnPeople()
totalPeople = len(personList)

#cv2.line(img,(0,0),(511,511),(255,0,0),5)
#cv2.circle(img,(50,50),18,(0,255,0),-1)



for i in range(1000):
    img = np.copy(map)
    #cv2.circle(img,(int(i/2),250),18,(0,255,0),-1)

    for person in personList:
        cv2.circle(img,(int(person.x),int(person.y)),int(r*person.radius)+2,(0,255,0),-1)
    
    cv2.imshow('display',img)
    cv2.waitKey(10)

    if i % 10 == 0:
        determineClosePeople()
    letPeopleMove()

    totalExited = sum([1 if p.exited else 0 for p in personList ])
    if totalExited == totalPeople:
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
