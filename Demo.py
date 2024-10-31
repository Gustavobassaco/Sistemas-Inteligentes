import pygame
from pygame.locals import * 
import sys
import random


class FlappyBird_Human:
    def __init__(self):
        pygame.font.init()
        self.screen = pygame.display.set_mode((400, 700))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 145
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 15
        self.gravity = 10
        self.dead = False
        self.dead2 = False
        self.sprite = 0
        self.counter = 0
        self.max = 0
        self.mortes = 0
        self.fit = 0
        self.offset = random.randint(-200, 200)
    
    def restart(self):
        self.gap = 145
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 15
        self.gravity = 10
        self.dead = False
        self.dead2 = False
        self.sprite = 0
        self.counter = 0
        self.fit = 0
        self.mortes = 0
        self.offset = random.randint(-200, 200)

    def birdCanoDistanceY(self):
        bird_center_y = self.bird.y + self.bird.height / 2
        downRect_y = 0 - self.gap - self.offset - 10 + self.wallDown.get_height()  # Coordenada y do cano inferior
        distance_y = bird_center_y - downRect_y
        return distance_y

    def shouldJump(self):
        if self.birdCanoDistanceY() >= 132: #125
            return True
        return False

    # Inputs da mlp
    def get_distances(self):
        bird_center_y = self.bird.y + self.bird.height / 2
        ceiling_distance = bird_center_y
        floor_distance = 700 - bird_center_y
        wall_distance_x = self.wallx - (self.bird.x + self.bird.width)
        
        upRect_y = 360 + self.gap - self.offset - 10 + self.wallDown.get_height()
        downRect_y = 0 - self.gap - self.offset - 10
        distance_to_top_pipe = self.bird.y - upRect_y
        distance_to_bottom_pipe = downRect_y - (self.bird.y + self.bird.height)

        return  distance_to_bottom_pipe, distance_to_top_pipe
        #return ceiling_distance ,distance_to_bottom_pipe

    def fitness(self):
        self.fit += self.counter
        return self.fit

    def updateWalls(self):
        self.wallx -= 4
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-200, 200)

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird):
            if not self.dead:
                self.mortes +=1
            self.dead = True
        if downRect.colliderect(self.bird):
            if not self.dead:
                self.mortes +=1
            self.dead = True
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 50
            self.dead = False
            self.dead2 = True
            #self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 10
            self.mortes +=1
        
    def run(self):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 10
                    self.jumpSpeed = 15

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                             (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                             (self.wallx, 0 - self.gap - self.offset))
            self.screen.blit(font.render(str(self.counter),
                                         -1,
                                         (255, 255, 255)),
                             (200, 50))
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
            
            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()
            if self.counter > self.max:
                self.max = self.counter
            pygame.display.update()

    def run_step(self, should_jump):

        self.fit += 0.01
        jump = self.shouldJump()
        if jump == should_jump:
            self.fit += 0.01

        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if should_jump and not self.dead:
            self.jump = 17 # 17
            self.gravity = 10
            self.jumpSpeed = 15 #15
        
        if self.dead:
            self.sprite = 2
        elif self.jump:
            self.sprite = 1
        self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))

        if not self.dead:
            self.sprite = 0

        self.updateWalls()
        self.birdUpdate()
        if self.counter > self.max:
            self.max = self.counter

    def run_stepG(self, should_jump):
        pygame.font.init()
        clock = pygame.time.Clock()
        clock.tick(0)
        font = pygame.font.SysFont("Arial", 50)
        self.fit += 0.01

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if should_jump and not self.dead:
            self.jump = 17
            self.gravity = 10
            self.jumpSpeed = 15
        
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.wallUp,
                            (self.wallx, 360 + self.gap - self.offset))
        self.screen.blit(self.wallDown,
                            (self.wallx, 0 - self.gap - self.offset))
        self.screen.blit(font.render(str(self.counter),
                                        -1,
                                        (255, 255, 255)),
                            (200, 50))
        if self.dead:
            self.sprite = 2
        elif self.jump:
            self.sprite = 1
        self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))

        if not self.dead:
            self.sprite = 0

        self.updateWalls()
        self.birdUpdate()
        if self.counter > self.max:
            self.max = self.counter
        pygame.display.update()

