import numpy as np
import random
import cv2
import queue
import pickle as pkl
import random
import scipy.spatial
from skimage.morphology import skeletonize

exitColor = [0,0,255]
baseColor = [0,0,0]
spawnColor = [0,255,0]
backgroundColor = [255,255,255]

DIRS = []

# starting right ccw
r2 = np.sqrt(2)
# DIRECTIONS = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])
# DIRECTION_LENGTHS = np.array([1, r2, 1, r2, 1, r2, 1, r2])
DIRECTIONS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
DIRECTION_LENGTHS = np.array([1, 1, 1, 1, r2, r2, r2, r2])
DIRECTION_ORDER = list(range(len(DIRECTIONS)))

DIRECTION_COLORS = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)]


r = 3
spawnRate = 0.02
collisionForceThreshold = 1.2
collisionInjury = .001
maxEmergencyPersonnel = 0
timeEmergencyPersonnel = 100 # how long before they spawn
centroidPosition = (0,0)

################################################################################
# Agent stuff
################################################################################

class Actor(object):
    def __init__(self, p):
        self.p = p
        self.closePeople = []
        self.color = (255,0,0)
        self.exited = False
        self.consecutiveFramesStuck = 0

    @property
    def x(self):
        return self.p[1]

    @x.setter
    def x(self, value):
        self.p[1] = value

    @property
    def y(self):
        return self.p[0]

    @y.setter
    def y(self, value):
        self.p[0] = value

    def draw(self, img):
        cv2.circle(img,(int(self.x), int(self.y)),2,self.color,-1)

    def move(self, personList):
        pass

class Beacon(Actor):
    def __init__(self, p, uuid):
        super().__init__(p)
        self.uuid = uuid
        self.color = (0, 0, 255)

    def draw(self, img):
        cv2.circle(img,(int(self.x), int(self.y)),1,self.color,-1)

class HomingBeacon(Beacon):
    def __init__(self, p, uuid, target):
        super().__init__(p, uuid)
        self.target = target

    def move(self, personList):
        for j in self.closePeople:
            otherPerson = personList[j]
            if isinstance(otherPerson, Person):
                if self.uuid not in otherPerson.homing_beacons:
                    otherPerson.plan = [self.target]
                    otherPerson.homing_beacons.add(self.uuid)

class MultiHomingBeacon(Beacon):
    def __init__(self, p, uuid, targets):
        super().__init__(p, uuid)
        self.targets = targets

    def move(self, personList):
        for j in self.closePeople:
            otherPerson = personList[j]
            if isinstance(otherPerson, Person):
                if self.uuid not in otherPerson.homing_beacons:
                    otherPerson.plan = [random.choice(self.targets)]
                    otherPerson.homing_beacons.add(self.uuid)

class AppBeacon(Beacon):
    def move(self, personList):
        for j in self.closePeople:
            otherPerson = personList[j]
            if isinstance(otherPerson, Person):
                otherPerson.app = True
                otherPerson.color = (0,0,255)

class Person(Actor):
    def __init__(self, p):
        super().__init__(p)

        self.speed = 0.5 + random.random()
        self.state = None

        self.radius = (0.25 + 0.5*random.random())
        self.collisions = 0
        self.injured = False
        self.homing_beacons = set()
        self.plan = []
        self.app = False
        self.leader = None
        if random.random() < 0.005:
            self.app = True
            self.color = (0,0,255)

    def move(self, personList):
        if self.exited:
            return

        x = int(self.x)
        y = int(self.y)

        shortestDir = np.array([0, 0])
        if self.app:
            if exit1Dirs[y,x] != 255:
                shortestDir = DIRECTIONS[exit1Dirs[y, x]]
        elif self.leader:
            xdir = self.x-self.leader.x
            ydir = self.y-self.leader.y
            dist = ((xdir)**2 + (ydir)**2)**(.5)
            if dist > 15:
                self.leader = None
            shortestDir = -np.array([ydir, xdir], np.float64)
            shortestDir /= np.linalg.norm(shortestDir) /2

        elif self.plan:
            ty, tx = self.plan[0]
            xdir = self.x-tx
            ydir = self.y-ty
            dist = ((xdir)**2 + (ydir)**2)**(.5)
            shortestDir = -np.array([ydir, xdir], np.float64)
            shortestDir /= np.linalg.norm(shortestDir) /2

        avoidWallsDir = np.array([0,0])
        if wallDirs[y][x] != 255:
            avoidWallsDir = DIRECTIONS[wallDirs[y][x]] / DIRECTION_LENGTHS[wallDirs[y][x]]
        wallDistance = wallDist[y][x]

        wallAvoidanceFactor = -1/(wallDistance+0.1)

        numCollisions = 0

        desiredX = 0
        desiredY = 0

        for j in self.closePeople:
            otherPerson = personList[j]
            if isinstance(otherPerson, Person):
                if not self.app and not self.leader and otherPerson.app:
                    self.leader = otherPerson

                xdir = (self.x-otherPerson.x)/r
                ydir = (self.y-otherPerson.y)/r
                dist = ((xdir)**2 + (ydir)**2)**(.5)
                # if dist > 2:
                #     continue
                desiredX += xdir / (dist+1)
                desiredY += ydir / (dist+1)
                if dist < self.radius + otherPerson.radius + 1:
                    self.collisions += 1
                    numCollisions += 1
                    if random.random() < collisionInjury:
                        self.injured = True

            # elif isinstance(otherPerson, HomingBeacon):
            #     if otherPerson.uuid not in self.homing_beacons:
            #         xdir = self.x-otherPerson.x
            #         ydir = self.y-otherPerson.y
            #         dist = ((xdir)**2 + (ydir)**2)**(.5)
            #         if dist < 5:
            #             self.homing_beacons.add(otherPerson.uuid)
            #         shortestDir = -np.array([ydir, xdir], np.float64)
            #         shortestDir /= np.linalg.norm(shortestDir)


        desiredDist = ((desiredX)**2 + (desiredY)**2)**(.5)
        if desiredDist > collisionForceThreshold:
            desiredX = collisionForceThreshold*desiredX/desiredDist
            desiredY = collisionForceThreshold*desiredY/desiredDist


        speedFactor = 1 / (numCollisions + 1)

        desiredX += self.speed * speedFactor * shortestDir[1] \
                            + wallAvoidanceFactor * avoidWallsDir[1]
        desiredY += self.speed * speedFactor * shortestDir[0] \
                            + wallAvoidanceFactor * avoidWallsDir[0]

        desiredX = self.x + r*desiredX/3
        desiredY = self.y + r*desiredY/3
        # try:
        newLoc = exit1Dist[int(desiredY)][int(desiredX)]
        if newLoc < 100000:
            self.x = desiredX
            self.y = desiredY
            self.consecutiveFramesStuck = 0
        else: # help out stuck people using teleportation
            self.consecutiveFramesStuck += 1
            if self.consecutiveFramesStuck > 5:
                desiredX = self.x + 15*(random.random()-0.5)
                desiredY = self.y + 15*(random.random()-0.5)
                newLoc = exit1Dist[int(desiredY)][int(desiredX)]
                if newLoc < 100000:
                    self.x = desiredX
                    self.y = desiredY
                    self.consecutiveFramesStuck = 0
            if self.consecutiveFramesStuck > 50:
                self.color = (0,255,255)

        if exit1Dist[y][x] < 2:
            self.exited = True


class Staff(Actor):
    def __init__(self, p):
        super().__init__(p)

        self.speed = 0.5 + random.random()
        self.state = None
        self.framesNotAllowedToMove = 0
        self.movingTowardsExit = False
        self.roamingDirection = (0,0)

        self.radius = (0.25 + 0.5*random.random())
        self.collisions = 0
        self.injured = False
        self.exited = False

        self.color = (255,255,0)

    def move(self, personList):
        self.framesNotAllowedToMove -= 1
        if self.framesNotAllowedToMove > 0:
            return

        x = int(self.x)
        y = int(self.y)

        curLoc = exit1Dist[y][x]
        if curLoc < 3:
            self.movingTowardsExit = False
            desiredX = self.x + 40*random.random() - 20
            desiredY = self.y + 40*random.random() - 20
            try:
                newLoc = exit1Dist[int(desiredY)][int(desiredX)]
                if newLoc < 100000 and newLoc > 3:
                    self.x = desiredX
                    self.y = desiredY
            except Exception:
                pass
            return

        shortestDir = np.array([0, 0])
        shortestDir = DIRECTIONS[exit1Dirs[y, x]]

        avoidWallsDir = np.array([0,0])
        if wallDirs[y][x] != 255:
            avoidWallsDir = DIRECTIONS[wallDirs[y][x]] / DIRECTION_LENGTHS[wallDirs[y][x]]
        wallDistance = wallDist[y][x]

        wallAvoidanceFactor = -7/(wallDistance+0.1)

        numCollisions = 0

        desiredX = 0
        desiredY = 0

        for j in self.closePeople:
            otherPerson = personList[j]
            if isinstance(otherPerson, Person):
                xdir = (self.x-otherPerson.x)/r
                ydir = (self.y-otherPerson.y)/r
                dist = ((xdir)**2 + (ydir)**2)**(.5)
                # if dist > 2:
                #     continue
                desiredX += xdir / (dist+1)
                desiredY += ydir / (dist+1)
                if dist < self.radius + otherPerson.radius + 1:
                    self.collisions += 1
                    numCollisions += 1
                    if random.random() < collisionInjury/10:
                        self.injured = True

        desiredDist = ((desiredX)**2 + (desiredY)**2)**(.5)
        if desiredDist > collisionForceThreshold:
            desiredX = collisionForceThreshold*desiredX/desiredDist
            desiredY = collisionForceThreshold*desiredY/desiredDist

        speedFactor = 1 / (numCollisions + 1)
        towardExit = 1 if self.movingTowardsExit else -1

        desiredX += self.speed * speedFactor *towardExit* shortestDir[1] \
                            + wallAvoidanceFactor * avoidWallsDir[1] \
                            + self.roamingDirection[1]
        desiredY += self.speed * speedFactor *towardExit* shortestDir[0] \
                            + wallAvoidanceFactor * avoidWallsDir[0] \
                            + self.roamingDirection[0]

        if random.random() < 0.01:
            self.roamingDirection = centroidPosition - self.p
            self.roamingDirection = self.roamingDirection / ((self.roamingDirection[0]**2+self.roamingDirection[1]**2)**(0.5))
            print("staff new roaming direction: " + str(self.roamingDirection))
        if random.random() < 0.002 and self.movingTowardsExit:
            self.movingTowardsExit = False
        if random.random() < 0.001 and not self.movingTowardsExit:
            self.movingTowardsExit = True


        desiredX = self.x + r*desiredX/3
        desiredY = self.y + r*desiredY/3
        try:
            newLoc = exit1Dist[int(desiredY)][int(desiredX)]
            if newLoc < 100000:
                self.x = desiredX
                self.y = desiredY
                self.consecutiveFramesStuck = 0
            else: # help out stuck agents using teleportation
                self.consecutiveFramesStuck += 1
                if self.consecutiveFramesStuck > 5:
                    desiredX = self.x + 30*(random.random()-0.5)
                    desiredY = self.y + 30*(random.random()-0.5)
                    newLoc = exit1Dist[int(desiredY)][int(desiredX)]
                    if newLoc < 100000:
                        self.x = desiredX
                        self.y = desiredY
                        self.consecutiveFramesStuck = 0
        except Exception:
            pass

# def spawnPeople(m):
#     personList = []
#     numEmergencyPersonnel = 0
#     for y in range(m.shape[0]):
#         for x in range(m.shape[1]):
#             if not (m[y,x]==backgroundColor).all():
#                 if random.random() < 0.001:
#                     p = Person(np.array([y, x]).astype(np.float64))
#                     personList.append(p)
            
#     return personList

def add_homing_chain(lst, chain, baseidx):
    for i in range(len(chain) - 1):
        lst.append(HomingBeacon(chain[i], baseidx+i, chain[i+1]))

def spawnPeople(map):
    personList = []
    numEmergencyPersonnel = 0
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            if (map[y][x]==spawnColor).all():
                if random.random() < spawnRate:
                    p = Person(np.array([y, x]).astype(np.float64))
                    personList.append(p)
            if (m[y,x]==exitColor).all():
                if numEmergencyPersonnel < maxEmergencyPersonnel:
                    s = Staff(np.array([y, x]).astype(np.float64))
                    personList.append(s)
                    numEmergencyPersonnel += 1
                    s.framesNotAllowedToMove = timeEmergencyPersonnel
                    print("staff spawned")

    personList.append(HomingBeacon(np.array([210, 290]).astype(np.float64),0, np.array([170, 314]).astype(np.float64)))
    personList.append(HomingBeacon(np.array([210, 340]).astype(np.float64),50, np.array([170, 314]).astype(np.float64)))

    chain = [
        np.array([170, 314]).astype(np.float64),
        np.array([150, 314]).astype(np.float64)
    ]

    add_homing_chain(personList, chain, 100)


    add_homing_chain(personList, [np.array([149, 314 + x* 40]).astype(np.float64) for x in range(1, 6)], 200)
    add_homing_chain(personList, [np.array([128, 314 -x* 40]).astype(np.float64) for x in range(1, 6)], 250)
    add_homing_chain(personList, [np.array([149, 314 -x* 40]).astype(np.float64) for x in range(1, 6)], 300)
    add_homing_chain(personList, [np.array([128, 314 +x* 40]).astype(np.float64) for x in range(1, 6)], 350)

    personList.append(MultiHomingBeacon(np.array([150, 314]).astype(np.float64), 400, [
        np.array([149, 314 + 40]).astype(np.float64),
        np.array([128, 314 - 40]).astype(np.float64),
        np.array([128, 314 + 40]).astype(np.float64),
        np.array([149, 314 - 40]).astype(np.float64)
    ]))

    personList.append(AppBeacon(np.array([149, 314 + 19 * 10]).astype(np.float64), 500))
    personList.append(AppBeacon(np.array([149, 314 - 19 * 10]).astype(np.float64), 600))
    personList.append(AppBeacon(np.array([128, 314 + 19 * 10]).astype(np.float64), 550))
    personList.append(AppBeacon(np.array([128, 314 - 19 * 10]).astype(np.float64), 660))

    chain = [
        np.array([250, 334]).astype(np.float64),
        np.array([270, 334]).astype(np.float64),
             np.array([320, 334]).astype(np.float64),
             np.array([320, 384]).astype(np.float64)
    ]

    add_homing_chain(personList, chain, 700)
    personList.append(AppBeacon(chain[-1], 800))

    chain = [
        np.array([250, 294]).astype(np.float64),
        np.array([270, 294]).astype(np.float64),
        np.array([320, 294]).astype(np.float64),
        np.array([320, 134]).astype(np.float64)
    ]

    add_homing_chain(personList, chain, 900)
    personList.append(AppBeacon(chain[-1], 1000))

    return personList

def determineClosePeople(map, personList):
    b = 20 # bin size
    distThresh = 2 # distance needed to be "close"
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

def computeDistanceToWalls(m):
    M, N, _ = m.shape
    grid = np.full((M, N), 100000.0)
    dirs = np.full((M, N), 255, np.uint8)
    gridimg = np.full((M, N, 3), 128, np.uint8)
    q = queue.Queue()

    for y in range(M):
        for x in range(N):
            if (m[y,x]==backgroundColor).all():
                grid[y,x] = 0
                gridimg[y,x,1] = 0
                gridimg[y,x,0] = 0
                p = np.array([y, x])
                for d in DIRECTIONS:
                    yp, xp = p + d
                    if yp >= 0 and xp >= 0 and yp < M and xp < N:
                        if (m[yp, xp]!=backgroundColor).any():
                            q.put(p + d)

    while q.qsize() > 0:
        y, x = p = q.get()

        if y < 1 or x < 1 or y > M-2 or x > N-2:
            continue

        best_dir, best_dist = -1, grid[y, x]
        random.shuffle(DIRECTION_ORDER)
        for i in DIRECTION_ORDER:
            d, l = DIRECTIONS[i], DIRECTION_LENGTHS[i]
            new_dist = grid[tuple(p + d)] + l
            if new_dist < best_dist:
                best_dir, best_dist = i, new_dist

        if best_dir != -1:
            dirs[y, x], grid[y, x] = best_dir, best_dist

            for d in DIRECTIONS:
                q.put(p + d)

        if dirs[y, x] < 8:
            gridimg[y, x] = DIRECTION_COLORS[dirs[y, x]]
        # gridimg[y,x,2] = min(255,int(grid[y, x]*3))

    #cv2.imshow('display', gridimg)
    #cv2.waitKey(0)
    return grid, dirs

def computeShortestPaths(m, wallDist):
    M, N, _ = m.shape
    grid = np.full((M, N), 100000.0)
    dirs = np.full((M, N), 255, np.uint8)
    gridimg = np.full((M, N, 3), 128, np.uint8)
    visited = np.full((M, N), 0)
    q = queue.Queue()

    for y in range(m.shape[0]):
        for x in range(m.shape[1]):
            if (m[y][x]==exitColor).all():
                gridimg[y][x][1] = 0
                gridimg[y][x][0] = 0
                grid[y][x] = 0
                p = np.array([y, x])
                for d in DIRECTIONS:
                    yp, xp = p + d
                    if yp >= 0 and xp >= 0 and yp < M and xp < N:
                        if (m[yp, xp]!=exitColor).any():
                            q.put(p + d)

    while q.qsize() > 0:
        y, x = p = q.get()

        if y < 1 or x < 1 or y > M-2 or x > N-2:
            continue

        if (m[y, x]==backgroundColor).all():
            continue
        if visited[y][x] > 0:
            continue
        visited[y][x] = 1

        wall_penalty = 2*max(0, 10-wallDist[y,x])
        
        best_dir, best_dist = -1, grid[y, x]
        random.shuffle(DIRECTION_ORDER)
        for i in DIRECTION_ORDER:
            d, l = DIRECTIONS[i], DIRECTION_LENGTHS[i]
            new_dist = grid[tuple(p + d)] + l #- 2*max(0, 10-wallDist[tuple(p + d)]) + wall_penalty
            if new_dist < best_dist:
                best_dir, best_dist = i, new_dist

        if best_dir != -1:
            dirs[y, x], grid[y, x] = best_dir, best_dist+wall_penalty

            for d in DIRECTIONS:
                q.put(p + d)

        if dirs[y, x] < 8:
            gridimg[y, x] = DIRECTION_COLORS[dirs[y, x]]
        # gridimg[y,x,2] = min(255,int(grid[y, x]*3))

    #cv2.imshow('display', gridimg)
    #cv2.waitKey(0)
    return grid, dirs

def cached_load(fname, thunk):
    try:
        pkl.load(open(fname, 'rb'))
    except:
        print(f'recomputing {fname}...')
        value = thunk()
        pkl.dump(value, open(fname, 'wb'))
        return value


################################################################################
# UTIL
################################################################################

def load_map(mapName):
    img = cv2.imread("maps/" + mapName + ".png", cv2.IMREAD_COLOR)
    return img

mapName = 'mona_lisa_rooms_2'
m = load_map(mapName)
people = spawnPeople(m)

# def skeleton(img):
#     img = np.copy(img)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#     done = False
#     size = np.size(img)
#     skel = np.zeros(img.shape,np.uint8)

#     while( not done):
#         eroded = cv2.erode(img,element)
#         temp = cv2.dilate(eroded,element)
#         temp = cv2.subtract(img,temp)
#         skel = cv2.bitwise_or(skel,temp)
#         img = eroded.copy()
    
#         zeros = size - cv2.countNonZero(img)
#         if zeros==size:
#             done = True
#     return skel
# _,_,rimg = cv2.split(m)
# _, dst = cv2.threshold(rimg, 250, 255, cv2.THRESH_BINARY)
# dst = cv2.bitwise_not(dst)
# skel = skeleton(dst)
# cv2.imshow('display',skel)
# cv2.waitKey(0)

try:
    wallDist = pkl.load(open('temp/' + mapName + 'wallDist2.pkl', 'rb'))
    wallDirs = pkl.load(open('temp/' + mapName + 'wallDirs2.pkl', 'rb'))
    exit1Dist = pkl.load(open('temp/' + mapName + 'exit1Dist2.pkl', 'rb'))
    exit1Dirs = pkl.load(open('temp/' + mapName + 'exit1Dirs2.pkl', 'rb'))
except Exception as e:
    print('recomputing distances...')
    wallDist, wallDirs = computeDistanceToWalls(m)
    exit1Dist, exit1Dirs = computeShortestPaths(m, wallDist)
    pkl.dump(wallDist,open('temp/' + mapName + 'wallDist2.pkl', 'wb'))
    pkl.dump(wallDirs,open('temp/' + mapName + 'wallDirs2.pkl', 'wb'))
    pkl.dump(exit1Dist,open('temp/' + mapName + 'exit1Dist2.pkl', 'wb'))
    pkl.dump(exit1Dirs,open('temp/' + mapName + 'exit1Dirs2.pkl', 'wb'))
    print('...done')

def main():
    cv2.namedWindow('display')
    actors = people

    for i in range(2000):
        img = np.copy(m)

        if i % 5 == 0:
            determineClosePeople(m, actors)

        centroidPosition = (0,0)
        for actor in actors:
            actor.move(actors)
            actor.draw(img)
            if not actor.exited:
                centroidPosition += actor.p
        centroidPosition /= len(actors)

        cv2.imshow('display',img)
        cv2.waitKey(10)

if __name__ == '__main__':
    main()