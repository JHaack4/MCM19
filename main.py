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

class Person:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.closePeople = []
        self.speed = 1 + random.random()
        self.exited = False

def letPeopleMove():

    for person in personList:
        if person.exited:
            continue
        x = int(person.x)
        y = int(person.y)
        shortestDir = dirsArray[exit1Dirs[y][x]]
        avoidWallsDir = dirsArray[wallDirs[y][x]]
        wallDistance = wallDist[y][x]
        wallAvoidanceFactor = -2/(wallDistance+1)

        numCollisions = 0
        desiredX = person.x
        desiredY = person.y
        
        for j in person.closePeople:
            otherPerson = personList[j]
            xdir = person.x-otherPerson.x
            ydir = person.y-otherPerson.y
            dist = ((xdir)**2 + (ydir)**2)**(.5)
            desiredX += xdir / (dist+1)
            desiredY += ydir / (dist+1)
            if dist < 5:
                numCollisions += 1
        
        speedFactor = 1 / (numCollisions + 1)

        desiredX += person.speed * speedFactor * shortestDir[1] \
                            + wallAvoidanceFactor * avoidWallsDir[1]
        desiredY += person.speed * speedFactor * shortestDir[0] \
                            + wallAvoidanceFactor * avoidWallsDir[0]

        

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
                if random.random() < 0.005:
                    p = Person(x,y)
                    personList.append(p)
                
def determineClosePeople():
    for i,p1 in enumerate(personList):
        if p1.exited:
            continue
        p1.closePeople = []
        for j,p2 in enumerate(personList):
            if i==j: continue
            if p2.exited: continue
            if ((p1.x-p2.x) ** 2 + (p1.y-p2.y) ** 2)**(.5) < 15:
                p1.closePeople.append(j)

################################################################################
# UTIL
################################################################################

def loadMap(mapName):
    img = cv2.imread("maps/" + mapName + ".png", cv2.IMREAD_COLOR)
    return img

# probably want to precompute/store these...
# create one for each exit

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
                    grid[y][x-1]+1, grid[y][x], grid[y-1][x-1]+1.4,
                    grid[y+1][x-1]+1.4, grid[y+1][x+1]+1.4, grid[y-1][x+1]+1.4]
        dirs[y][x] = np.argmin(a)
        grid[y][x] = min(a)
        
        q.put((y-1,x))
        q.put((y+1,x))
        q.put((y,x+1))
        q.put((y,x-1))
        gridimg[y][x][2] = min(255,int(grid[y][x]/1.5))

    #cv2.imshow('display', gridimg)
    #cv2.waitKey(0)
    return grid, dirs

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
        cv2.circle(img,(int(person.x),int(person.y)),5,(0,255,0),-1)
    
    cv2.imshow('display',img)
    cv2.waitKey(10)

    if i % 5 == 0:
        determineClosePeople()
    letPeopleMove()

    totalExited = sum([1 if p.exited else 0 for p in personList ])


cv2.waitKey(0)
cv2.destroyAllWindows()
