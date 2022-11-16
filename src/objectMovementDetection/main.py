
import math
import pygame
from sys import exit
import random
from const import *
from utils import *
from table import *


class Player():
    def __init__(self, maxVelocity, maxRotationVelocity):
        super().__init__()
        global GLOBAL_SCREEN
        self.xPos, self.yPos = PlayerParam.INITIAL_X  + random.randint(-int(0.3*PlayerParam.INITIAL_X), int(
            0.3*PlayerParam.INITIAL_X)), PlayerParam.INITIAL_Y

        self.maxVelocity = maxVelocity
        self.maxRotationVelocity = maxRotationVelocity

        self.currVelocity = PlayerParam.ACCELERATION_FORWARD  # always >= 0
        self.currRotationVelocity = 0  # rotate left < 0, rotate right > 0
        self.currAngle = math.pi
        # self.accelerationForward = PlayerParam.ACCELERATION_FORWARD

        self.circleRect = pygame.draw.circle(
            GLOBAL_SCREEN, CustomColor.RED, (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

        # Raycasting
        self.rayCastingLists = [PlayerParam.INFINITY] * PlayerParam.CASTED_RAYS

        self.mode = MODE_PLAY.I_AM_A_ROBOT
        # self.mode = MODE_PLAY.RL_TRAIN
        # self.mode = MODE_PLAY.MANUAL
        self.displayGUI = GUI.DISPLAY

        self.memAction = dict()

        # FOR DEPLOY MODE
        self.deployedQTabled = None

    def loadQTable(self):
        if (self.mode == MODE_PLAY.RL_DEPLOY):
            file = open(FILE.MODEL_SAVE, "r+")
            RLInFile = file.read()
            self.deployedQTabled = json.loads(RLInFile)
            file.close()

    def _move(self):
        dt = float(1/GameSettingParam.FPS)
        
        self.currAngle += (self.currRotationVelocity*dt) 
        if (self.currAngle < 0):
            self.currAngle = 2*math.pi - abs(self.currAngle)
        elif (self.currAngle > 2*math.pi):
            self.currAngle = abs(self.currAngle - 2*math.pi)
            
        self.yPos += math.cos(self.currAngle) * self.currVelocity * dt
        self.xPos += -math.sin(self.currAngle) * self.currVelocity * dt        

    def _playerInput(self, actionIndex=None):
        if (self.mode == MODE_PLAY.MANUAL):
            keys = pygame.key.get_pressed()

            # Rotate left ()
            if keys[pygame.K_a]:
                self.currRotationVelocity -= PlayerParam.ACCELERATION_ROTATE
            # Rotate right ()
            if keys[pygame.K_d]:
                self.currRotationVelocity += PlayerParam.ACCELERATION_ROTATE

            # Increase forward velocity
            if keys[pygame.K_w]:
                self.currVelocity = min(
                    self.currVelocity + PlayerParam.ACCELERATION_FORWARD, self.maxVelocity)

            # Stop
            if keys[pygame.K_s]:
                self.currVelocity = 0
                self.currRotationVelocity = 0

            # Decrease forward velocity
            if keys[pygame.K_x]:
                self.currVelocity = max(
                    self.currVelocity - PlayerParam.ACCELERATION_FORWARD, 0)

        elif (self.mode == MODE_PLAY.RL_TRAIN):

            if RLParam.ACTIONS[actionIndex] == PlayerParam.DESC_ROTATION_VELO:
                self.currRotationVelocity -= PlayerParam.ACCELERATION_ROTATE

            if RLParam.ACTIONS[actionIndex] == PlayerParam.INC_ROTATION_VELO:
                self.currRotationVelocity += PlayerParam.ACCELERATION_ROTATE

            if RLParam.ACTIONS[actionIndex] == PlayerParam.STOP:
                self.currVelocity = 0
                self.currRotationVelocity = 0

            if RLParam.ACTIONS[actionIndex] == PlayerParam.INC_FORWARD_VELO:
                self.currVelocity = min(
                    self.currVelocity + PlayerParam.ACCELERATION_FORWARD, self.maxVelocity)

            if RLParam.ACTIONS[actionIndex] == PlayerParam.DESC_FORWARD_VELO:
                self.currVelocity = max(
                    self.currVelocity - PlayerParam.ACCELERATION_FORWARD, 0)

        elif (self.mode == MODE_PLAY.RL_DEPLOY):
            currentState = RLAlgorithm.hashFromDistanceToState(signalPerAreaData=RLAlgorithm.convertRayCastingDataToSignalPerArea(rayCastingData=self.rayCastingLists),
                                                               leftSideDistance=abs(
                                                                   self.xPos),
                                                               rightSideDistance=abs(
                                                                   self.xPos - GameSettingParam.WIDTH),
                                                               angle=self.currAngle,
                                                               RotationVelocity=self.currRotationVelocity,
                                                               yVelo=- self.currVelocity * math.cos(self.currAngle),)

            decidedAction = np.argmax(self.deployedQTabled[currentState])

            if RLParam.ACTIONS[decidedAction] == PlayerParam.DESC_FORWARD_VELO:
                self.currRotationVelocity -= PlayerParam.ACCELERATION_ROTATE

            if RLParam.ACTIONS[decidedAction] == PlayerParam.INC_FORWARD_VELO:
                self.currRotationVelocity += PlayerParam.ACCELERATION_ROTATE

            if RLParam.ACTIONS[decidedAction] == PlayerParam.STOP:
                self.currVelocity = 0
                self.currRotationVelocity = 0

            if RLParam.ACTIONS[decidedAction] == PlayerParam.INC_FORWARD_VELO:
                self.currVelocity = min(
                    self.currVelocity + PlayerParam.ACCELERATION_FORWARD, self.maxVelocity)

            if RLParam.ACTIONS[decidedAction] == PlayerParam.DESC_FORWARD_VELO:
                self.currVelocity = max(
                    self.currVelocity - PlayerParam.ACCELERATION_FORWARD, 0)
            return decidedAction

        elif (self.mode == MODE_PLAY.I_AM_A_ROBOT):
            currentState = RLAlgorithm.hashFromDistanceToState(signalPerAreaData=RLAlgorithm.convertRayCastingDataToSignalPerArea(rayCastingData=self.rayCastingLists),
                                                               leftSideDistance=abs(
                                                                   self.xPos),
                                                               rightSideDistance=abs(
                                                                   self.xPos - GameSettingParam.WIDTH),
                                                               angle=self.currAngle,
                                                               RotationVelocity=self.currRotationVelocity,
                                                               yVelo=- self.currVelocity * math.cos(self.currAngle),)

            # decidedAction = np.argmax(self.deployedQTabled[currentState])

            if currentState not in self.memAction.keys():
                print("state:",currentState,"is unknow, decide a action now.")
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.KEYDOWN:
                                    # Rotate left ()
                        if event.key == pygame.K_a:
                            self.memAction[currentState] = 1
                        elif event.key == pygame.K_d:
                            self.memAction[currentState] = 0
                        elif event.key == pygame.K_w:
                            self.memAction[currentState] = 3
                            # print("here")
                        elif event.key == pygame.K_s:
                            self.memAction[currentState] = 2
                        elif event.key == pygame.K_x:
                            self.memAction[currentState] = 4
                        elif event.key == pygame.K_SPACE:
                            self.memAction[currentState] = 5

                        print("you decide state",currentState," is action ",self.memAction[currentState])
                        break

            

            decidedAction = self.memAction[currentState]

            if RLParam.ACTIONS[decidedAction] == PlayerParam.DESC_ROTATION_VELO:
                self.currRotationVelocity -= PlayerParam.ACCELERATION_ROTATE
                
            elif RLParam.ACTIONS[decidedAction] == PlayerParam.INC_ROTATION_VELO:
                self.currRotationVelocity += PlayerParam.ACCELERATION_ROTATE

            elif RLParam.ACTIONS[decidedAction] == PlayerParam.STOP:
                self.currVelocity = 0
                self.currRotationVelocity = 0

            elif RLParam.ACTIONS[decidedAction] == PlayerParam.INC_FORWARD_VELO:
                self.currVelocity = min(
                    self.currVelocity + PlayerParam.ACCELERATION_FORWARD, self.maxVelocity)
                # print("go here")

            elif RLParam.ACTIONS[decidedAction] == PlayerParam.DESC_FORWARD_VELO:
                self.currVelocity = max(
                    self.currVelocity - PlayerParam.ACCELERATION_FORWARD, 0)
            
            elif RLParam.ACTIONS[decidedAction] == PlayerParam.DO_NOTHING_VELO:
                pass


    def _rayCasting(self):
        # print("  ///")
        # startTime = time.time()
        global obstacles
        inRangedObj = []

        # stop = False

        for obstacle in obstacles:
            if Utils.distanceBetweenTwoPoints(self.xPos, self.yPos, obstacle.xPos, obstacle.yPos) < PlayerParam.RADIUS_LIDAR + PlayerParam.RADIUS_OBJECT:
                inRangedObj.append(obstacle)
        startAngle = self.currAngle - PlayerParam.HALF_FOV

        if len(inRangedObj) == 0:
            for ray in range(PlayerParam.CASTED_RAYS):
                target_x = self.xPos - \
                    math.sin(startAngle) * PlayerParam.RADIUS_LIDAR
                target_y = self.yPos + \
                    math.cos(startAngle) * PlayerParam.RADIUS_LIDAR
                pygame.draw.line(GLOBAL_SCREEN, CustomColor.GREEN,
                                 (self.xPos, self.yPos), (target_x, target_y))
                self.rayCastingLists[ray] = PlayerParam.INFINITY
                startAngle += PlayerParam.STEP_ANGLE
        else:
            for ray in range(PlayerParam.CASTED_RAYS):
                # get ray target coordinates
                isDetectObject = False

                for obstacle in inRangedObj:
                    theda = math.sqrt((obstacle.xPos - self.xPos)
                                      ** 2+(obstacle.yPos - self.yPos)**2)
                    beta = 0
                    if obstacle.xPos - self.xPos < 0:
                        beta = math.acos(
                            (obstacle.yPos - self.yPos)/theda) - startAngle
                    else:
                        beta = math.acos(
                            (obstacle.yPos - self.yPos)/theda) + startAngle
                    height = theda*math.sin(beta)

                    tvh = (self.xPos-obstacle.xPos)*math.sin(startAngle) + (obstacle.yPos-self.yPos)*math.cos(startAngle)
                    if abs(height) < PlayerParam.RADIUS_OBJECT and tvh > 0:
                        isDetectObject = True

                if not isDetectObject:
                    self.rayCastingLists[ray] = PlayerParam.INFINITY
                    target_x = self.xPos - \
                        math.sin(startAngle) * PlayerParam.RADIUS_LIDAR
                    target_y = self.yPos + \
                        math.cos(startAngle) * PlayerParam.RADIUS_LIDAR
                    pygame.draw.line(GLOBAL_SCREEN, CustomColor.RED,
                                     (self.xPos, self.yPos), (target_x, target_y))
                    self.rayCastingLists[ray] = PlayerParam.INFINITY
                    startAngle += PlayerParam.STEP_ANGLE
                    continue

                isDetectObject = False

                for depth in range(0, PlayerParam.RADIUS_LIDAR + 1, 10):
                    target_x = self.xPos - \
                        math.sin(startAngle) * depth
                    target_y = self.yPos + \
                        math.cos(startAngle) * depth
                        
                    for obstacle in inRangedObj:
                        distance = Utils.distanceBetweenTwoPoints(
                            target_x, target_y, obstacle.xPos, obstacle.yPos)
                        if distance <= PlayerParam.RADIUS_OBJECT:
                            self.rayCastingLists[ray] = Utils.distanceBetweenTwoPoints(
                                target_x, target_y, self.xPos, self.yPos)
                            # stop = True
                            isDetectObject = True
                            pygame.draw.line(
                                GLOBAL_SCREEN, CustomColor.WHITE, (self.xPos, self.yPos), (target_x, target_y))
                        if depth == PlayerParam.RADIUS_LIDAR and not isDetectObject:
                            self.rayCastingLists[ray] = PlayerParam.INFINITY
                            # if self.displayGUI == GUI.DISPLAY:
                            pygame.draw.line(
                                GLOBAL_SCREEN, CustomColor.PINK, (self.xPos, self.yPos), (target_x, target_y))
                        if isDetectObject:
                            break
                startAngle += PlayerParam.STEP_ANGLE

    def checkCollision(self):
        global player, obstacles
        for obstacle in obstacles:
            distanceBetweenCenter = Utils.distanceBetweenTwoPoints(
                self.xPos, self.yPos, obstacle.xPos, obstacle.yPos)
            # https://stackoverflow.com/questions/22135712/pygame-collision-detection-with-two-circles
            if distanceBetweenCenter <= 2*PlayerParam.RADIUS_OBJECT or self.yPos < 0:
                # print("Ouch!!!")
                player = Player(maxVelocity=PlayerParam.MAX_VELOCITY,
                                maxRotationVelocity=PlayerParam.MAX_ROTATION_VELOCITY)
                obstacles = []
                for _ in range(ObstacleParam.NUMBER_OF_OBSTACLES):
                    obstacles.append(Obstacle())
                return True
        return False

    def draw(self, actionIndex):
        global GLOBAL_SCREEN
        tempActionIndex = self._playerInput(actionIndex=actionIndex)
        self._rayCasting()
        self.checkCollision()
        self._move()

        if (self.displayGUI == GUI.DISPLAY):
            # draw player on 2D board
            pygame.draw.circle(GLOBAL_SCREEN, CustomColor.RED,
                               (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

            # draw player direction
            pygame.draw.line(GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos),
                             (self.xPos - math.sin(self.currAngle) * 20,
                              self.yPos + math.cos(self.currAngle) * 20), 3)
            return tempActionIndex

    def I_AM_A_ROBOT(self, actionIndex):
        global GLOBAL_SCREEN
        
        self._playerInput(actionIndex=actionIndex)
        self._rayCasting()
        self.checkCollision()
        self._move()

        if (self.displayGUI == GUI.DISPLAY):
            # draw player on 2D board
            pygame.draw.circle(GLOBAL_SCREEN, CustomColor.RED,
                               (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

            # draw player direction
            pygame.draw.line(GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos),
                             (self.xPos - math.sin(self.currAngle) * 20,
                              self.yPos + math.cos(self.currAngle) * 20), 3)


class Obstacle(Player):
    def __init__(self):
        super().__init__(maxVelocity=PlayerParam.MAX_VELOCITY,
                         maxRotationVelocity=PlayerParam.MAX_ROTATION_VELOCITY)

        self.xPos = ObstacleParam.INITIAL_OBSTACLE_X + random.randint(-int(0.8*ObstacleParam.INITIAL_OBSTACLE_X), int(
            0.8*ObstacleParam.INITIAL_OBSTACLE_X))

        self.yPos = ObstacleParam.INITIAL_OBSTACLE_Y + random.randint(0, int(
            0.7*GameSettingParam.HEIGHT))

        self.circleRect = pygame.draw.circle(
            GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)
        self.currAngle = 0
        self.maxVelocity = ObstacleParam.MAX_VELOCITY

        # Is random ?
        self.randomVelo = False

    def _loadQTable(self):
        pass

    def _playerInput(self):
        keys = RLParam.ACTIONS
        probs = ObstacleParam.PROBABILITIES_ACTION

        # choosedKey = keys[random.choice(range(len(keys)))]
        randomIndex = random.choices(range(len(keys)), probs)[0]
        choosedKey = keys[randomIndex]

        if choosedKey == PlayerParam.INC_ROTATION_VELO:
            self.currRotationVelocity += ObstacleParam.OBSTACLE_ACCELERATION_ROTATE
        if choosedKey == PlayerParam.DESC_ROTATION_VELO:
            self.currRotationVelocity -= ObstacleParam.OBSTACLE_ACCELERATION_ROTATE
        if choosedKey == PlayerParam.STOP:
            self.currVelocity = 0
            self.currRotationVelocity = 0
        if choosedKey == PlayerParam.INC_FORWARD_VELO:
            self.currVelocity = min(
                self.currVelocity + ObstacleParam.OBSTACLE_ACCELERATION_FORWARD, self.maxVelocity)
        if choosedKey == PlayerParam.DESC_FORWARD_VELO:
            self.currVelocity = max(
                self.currVelocity - ObstacleParam.OBSTACLE_ACCELERATION_FORWARD, 0)

    def _rayCasting(self):
        pass

    def checkCollision(self):
        pass

    def draw(self):
        global GLOBAL_SCREEN
        self._playerInput()
        self._move()

        if (self.displayGUI == GUI.DISPLAY):
            # draw player on 2D board
            pygame.draw.circle(GLOBAL_SCREEN, CustomColor.GREEN,
                               (self.xPos, self.yPos), PlayerParam.RADIUS_OBJECT)

            pygame.draw.circle(GLOBAL_SCREEN, CustomColor.RED,
                               (self.xPos, self.yPos), 6)
            # draw player direction
            pygame.draw.line(GLOBAL_SCREEN, CustomColor.GREEN, (self.xPos, self.yPos),
                             (self.xPos - math.sin(self.currAngle) * 20,
                              self.yPos + math.cos(self.currAngle) * 20), 3)


class Environment:
    def __init__(self, currentPlayer, currentObstacles, modeGUI=GUI.HIDDEN):
        self.currPlayer = currentPlayer
        self.currObstacles = currentObstacles

        self.rayCastingData = currentPlayer.rayCastingLists
        self.xPos, self.yPos = currentPlayer.xPos, currentPlayer.yPos

        self.previousYPos, self.previousXPos, self.previousAngle, self.previousVelocity = self.yPos, self.xPos, self.currPlayer.currAngle, self.currPlayer.currVelocity

        self.modeGUI = modeGUI
        
        currentPlayer.mode = MODE_PLAY.RL_TRAIN
        currentPlayer.displayGUI = modeGUI

        for obstacle in currentObstacles:
            obstacle.mode = MODE_PLAY.RL_TRAIN
            obstacle.displayGUI = modeGUI

    def _isDoneEpisode(self):
        return \
            self.yPos <= 0 or self.yPos > GameSettingParam.HEIGHT or \
            self.xPos <= 0 or self.xPos >= GameSettingParam.WIDTH or \
            self.currPlayer.currAngle < RLParam.LEVEL_OF_ANGLE.OVER_ROTATION_LEFT_ANGLE or\
            self.currPlayer.currAngle > RLParam.LEVEL_OF_ANGLE.OVER_ROTATION_RIGHT_ANGLE or\
            self.currPlayer.checkCollision()

    def _selfUpdated(self):
        self.rayCastingData = self.currPlayer.rayCastingLists
        self.xPos, self.yPos = self.currPlayer.xPos, self.currPlayer.yPos

    def updateStateByAction(self, actionIndex):
        # Use when want to visualize when training
        if (self.currPlayer.displayGUI == GUI.DISPLAY):
            GLOBAL_CLOCK.tick(GameSettingParam.FPS)
            GLOBAL_SCREEN.fill(CustomColor.BLACK)
            GLOBAL_SCREEN.blit(GLOBAL_SCREEN, (0, 0))            

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                    
        for obstacle in obstacles:
            obstacle.draw()
        self.previousYPos, self.previousXPos, self.previousAngle, self.previousVelocity = self.yPos, self.previousXPos, self.currPlayer.currAngle, self.currPlayer.currVelocity
        self.currPlayer.draw(actionIndex=actionIndex)
        self._selfUpdated()

        nextState = RLAlgorithm.hashFromDistanceToState(
            signalPerAreaData=RLAlgorithm.convertRayCastingDataToSignalPerArea(
                rayCastingData=self.rayCastingData),
            leftSideDistance=abs(self.xPos),
            rightSideDistance=abs(self.xPos - GameSettingParam.WIDTH),
            angle=self.currPlayer.currAngle,
            RotationVelocity=self.currPlayer.currRotationVelocity,
            yVelo=- self.currPlayer.currVelocity * math.cos(self.currPlayer.currAngle),)

        reward = RLAlgorithm.getReward(
            currState=nextState,
            previousInfo={
                "xPos": self.previousXPos,
                "yPos": self.previousYPos,
                "velocity": self.previousVelocity,
                "angle": self.previousAngle
            },
            currentInfo={
                "xPos": self.xPos,
                "yPos": self.yPos,
                "velocity": self.currPlayer.currVelocity,
                "angle": self.currPlayer.currAngle
            },
            actionIndex=actionIndex)

        done = self._isDoneEpisode()
        # Use when want to visualize when training
        if (self.currPlayer.displayGUI == GUI.DISPLAY):
            pygame.display.flip()

        return nextState, reward, done

    def getCurrentState(self):
        return RLAlgorithm.hashFromDistanceToState(signalPerAreaData=RLAlgorithm.convertRayCastingDataToSignalPerArea(rayCastingData=self.rayCastingData),
                                                   leftSideDistance=abs(
                                                       self.xPos),
                                                   rightSideDistance=abs(
                                                       self.xPos - GameSettingParam.WIDTH),
                                                   angle=self.currPlayer.currAngle,
                                                   RotationVelocity=self.currPlayer.currRotationVelocity,
                                                   yVelo=- self.currPlayer.currVelocity * math.cos(self.currPlayer.currAngle),)

    def reset(self):
        currGUIMode = self.modeGUI
        del self
        global player, obstacles
        player = Player(maxVelocity=PlayerParam.MAX_VELOCITY,
                        maxRotationVelocity=PlayerParam.MAX_ROTATION_VELOCITY)
        obstacles = []
        for _ in range(ObstacleParam.NUMBER_OF_OBSTACLES):
            obstacles.append(Obstacle())
        return Environment(currentPlayer=player, currentObstacles=obstacles, modeGUI=currGUIMode)

###########################################################################################


# Game setting
pygame.init()
GLOBAL_SCREEN = pygame.display.set_mode(
    (GameSettingParam.WIDTH, GameSettingParam.HEIGHT))
pygame.display.set_caption(GameSettingParam.CAPTION)
GLOBAL_CLOCK = pygame.time.Clock()

# Groups
player = Player(maxVelocity=PlayerParam.MAX_VELOCITY,
                maxRotationVelocity=PlayerParam.MAX_ROTATION_VELOCITY)
obstacles = []
for _ in range(ObstacleParam.NUMBER_OF_OBSTACLES):
    obstacles.append(Obstacle())


def startGame(mode=MODE_PLAY.MANUAL):
    if (mode == MODE_PLAY.MANUAL) or (mode == MODE_PLAY.I_AM_A_ROBOT):
        while True:
            GLOBAL_CLOCK.tick(GameSettingParam.FPS)
            GLOBAL_SCREEN.fill(CustomColor.BLACK)
            GLOBAL_SCREEN.blit(GLOBAL_SCREEN, (0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            player.draw(actionIndex=None)
            for obstacle in obstacles:
                obstacle.draw()

            pygame.display.flip()

    elif (mode == MODE_PLAY.RL_TRAIN):
        env = Environment(currentPlayer=player, currentObstacles=obstacles, modeGUI=GUI.DISPLAY)
        RL = RLAlgorithm(rayCastingData=env.rayCastingData,
                         actions=RLParam.ACTIONS)
        RL.train(env)
    elif (mode == MODE_PLAY.RL_DEPLOY):
        player.mode = MODE_PLAY.RL_DEPLOY
        player.loadQTable()
        while True:
            GLOBAL_CLOCK.tick(GameSettingParam.FPS)
            GLOBAL_SCREEN.fill(CustomColor.BLACK)
            GLOBAL_SCREEN.blit(GLOBAL_SCREEN, (0, 0))
            
            # keys = pygame.key.get_pressed()
            # if (keys[pygame.K_SPACE]):
            #     pygame.time.wait(15000)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            actionIndex = player.draw(actionIndex=None)
            print(f"==========> actionIndex {actionIndex}")
            for obstacle in obstacles:
                obstacle.draw()

            pygame.display.flip()
            
            ray = RLAlgorithm.convertRayCastingDataToSignalPerArea(rayCastingData=player.rayCastingLists)
            currentState = RLAlgorithm.hashFromDistanceToState(signalPerAreaData=ray,
                                                               leftSideDistance=abs(
                                                                   player.xPos),
                                                               rightSideDistance=abs(
                                                                   player.xPos - GameSettingParam.WIDTH),
                                                               angle=player.currAngle,
                                                               RotationVelocity=player.currRotationVelocity,
                                                               yVelo= -player.currVelocity * math.cos(player.currAngle),)
            print(f"with lidar: [{ray[0]}, {ray[1]}, {ray[2]}, {ray[3]}], leftSide: {abs(player.xPos)}, angle: {player.currAngle}, RotationVelocity: {player.currRotationVelocity}, yVelo: {-player.currVelocity * math.cos(player.currAngle)}",)
            # print("==> gain state {}", currentState)
            
            reward = RLAlgorithm.getReward(currState=currentState, actionIndex=actionIndex, previousInfo=5, currentInfo=4)
            
            print()
            print(f"move {RLParam.ACTIONS[actionIndex]} then gain {reward} scores")
            print("\n\n")
            
            pygame.time.wait(100)



startGame(mode=MODE_PLAY.RL_TRAIN)
# startGame(mode=MODE_PLAY.MANUAL)
# startGame(mode=MODE_PLAY.RL_DEPLOY)
# startGame(mode=MODE_PLAY.I_AM_A_ROBOT)
