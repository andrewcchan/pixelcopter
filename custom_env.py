import math
import sys
import pygame
import numpy as np
from ple.games.base.pygamewrapper import PyGameWrapper
from ple.games.utils.vec2d import vec2d
from pygame.constants import K_w

class Block(pygame.sprite.Sprite):
    def __init__(self, pos_init, speed, width, height, color=(120, 240, 80)):
        pygame.sprite.Sprite.__init__(self)
        self.pos = vec2d(pos_init)
        self.speed = speed
        self.width = width
        self.height = height

        image = pygame.Surface((self.width, self.height))
        image.fill(color)
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = pos_init

    def update(self, dt):
        self.pos.x -= self.speed * dt
        self.rect.topleft = (self.pos.x, self.pos.y)

class Trap(pygame.sprite.Sprite):
    def __init__(self, pos_init, speed, width, height, move_y=False):
        pygame.sprite.Sprite.__init__(self)
        self.pos = vec2d(pos_init)
        self.start_y = pos_init[1]
        self.speed = speed
        self.width = width
        self.height = height
        self.move_y = move_y
        self.time = 0

        image = pygame.Surface((self.width, self.height))
        image.fill((255, 0, 0)) # Red for traps
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = pos_init

    def update(self, dt):
        self.pos.x -= self.speed * dt
        if self.move_y:
            self.time += dt
            # Move up and down
            self.pos.y = self.start_y + math.sin(self.time * 0.005) * (self.height * 2)
        self.rect.topleft = (self.pos.x, self.pos.y)

class Goal(pygame.sprite.Sprite):
    def __init__(self, pos_init, speed, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.pos = vec2d(pos_init)
        self.speed = speed
        self.width = width
        self.height = height

        image = pygame.Surface((self.width, self.height))
        image.fill((0, 255, 255)) # Cyan for goal
        # Create a transparent surface for the goal gate effect
        image.set_alpha(128)
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = pos_init

    def update(self, dt):
        self.pos.x -= self.speed * dt
        self.rect.topleft = (self.pos.x, self.pos.y)

class HelicopterPlayer(pygame.sprite.Sprite):
    def __init__(self, speed, SCREEN_WIDTH, SCREEN_HEIGHT):
        pygame.sprite.Sprite.__init__(self)
        # Start position
        pos_init = (int(SCREEN_WIDTH * 0.2), SCREEN_HEIGHT / 2)
        self.pos = vec2d(pos_init)
        self.speed = speed
        self.climb_speed = speed * -0.875
        self.fall_speed = speed * 0.09
        self.momentum = 0
        self.width = SCREEN_WIDTH * 0.05
        self.height = SCREEN_HEIGHT * 0.05

        image = pygame.Surface((self.width, self.height))
        image.fill((255, 255, 255))
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, is_climbing, dt):
        self.momentum += (self.climb_speed if is_climbing else self.fall_speed) * dt
        self.momentum *= 0.99
        self.pos.y += self.momentum
        self.rect.center = (self.pos.x, self.pos.y)

class CustomPixelcopter(PyGameWrapper):
    def __init__(self, width=256, height=256):
        actions = {"up": K_w}
        PyGameWrapper.__init__(self, width, height, actions=actions)
        self.speed = 0.0004 * width
        self.level = 1
        self.max_levels = 5
        self.allowed_fps = 30

        self.rewards = {
            "positive": 1.0,
            "negative": -1.0,
            "tick": 0.01, # Small reward for surviving
            "loss": -5.0,
            "win": 10.0
        }

    def init(self):
        self.score = 0.0
        self.lives = 1.0
        self.player = HelicopterPlayer(self.speed, self.width, self.height)
        self.player_group = pygame.sprite.Group(self.player)
        self.block_group = pygame.sprite.Group()
        self.trap_group = pygame.sprite.Group()
        self.goal_group = pygame.sprite.Group()

        self.load_level()

    def load_level(self):
        self.block_group.empty()
        self.trap_group.empty()
        self.goal_group.empty()

        # Determine level parameters
        level_length = 2000 + self.level * 500
        current_x = self.width * 1.5

        while current_x < level_length:
            # Distance between obstacles
            dist_next = self.rng.randint(int(self.width * 0.8), int(self.width * 1.5)) - (self.level * 10)
            dist_next = max(dist_next, int(self.width * 0.4))

            current_x += dist_next

            # Choose obstacle type
            if self.rng.rand() < 0.7: # Mostly blocks
                y = self.rng.randint(0, self.height - int(self.height*0.3))
                h = self.rng.randint(int(self.height*0.1), int(self.height*0.4))
                w = int(self.width * 0.05)
                self.block_group.add(Block((current_x, y), self.speed, w, h))
            else: # trap
                y = self.rng.randint(int(self.height*0.1), self.height - int(self.height*0.1))
                h = int(self.height * 0.05)
                w = int(self.width * 0.05)
                move_y = self.level >= 2
                self.trap_group.add(Trap((current_x, y), self.speed, w, h, move_y))

        # Add Goal at the end
        self.goal_group.add(Goal((level_length + self.width * 0.5, 0), self.speed, int(self.width*0.1), self.height))

    def getGameState(self):
        # Find closest object
        next_obj_dist = 9999
        next_obj_top = 0
        next_obj_bottom = 0
        obj_type = 0 # 0: none, 1: block, 2: trap, 3: goal

        # Check blocks
        for b in self.block_group:
            dist = b.pos.x - self.player.pos.x
            if 0 < dist < next_obj_dist:
                next_obj_dist = dist
                next_obj_top = b.pos.y
                next_obj_bottom = b.pos.y + b.height
                obj_type = 1

        # Check traps
        for t in self.trap_group:
            dist = t.pos.x - self.player.pos.x
            if 0 < dist < next_obj_dist:
                next_obj_dist = dist
                next_obj_top = t.pos.y
                next_obj_bottom = t.pos.y + t.height
                obj_type = 2

        # Check goal
        for g in self.goal_group:
            dist = g.pos.x - self.player.pos.x
            if 0 < dist < next_obj_dist:
                next_obj_dist = dist
                next_obj_top = 0
                next_obj_bottom = self.height
                obj_type = 3

        state = {
            "player_y": self.player.pos.y,
            "player_vel": self.player.momentum,
            "next_obj_dist": next_obj_dist,
            "next_obj_top": next_obj_top,
            "next_obj_bottom": next_obj_bottom,
            "obj_type": obj_type,
            "level": self.level
        }
        return state

    def step(self, dt):
        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        self.player.update(self.is_climbing, dt)
        self.block_group.update(dt)
        self.trap_group.update(dt)
        self.goal_group.update(dt)

        # Check collisions
        if pygame.sprite.spritecollideany(self.player, self.block_group) or \
           pygame.sprite.spritecollideany(self.player, self.trap_group):
            self.lives = -1
            self.score += self.rewards["loss"]

        # Check boundary
        if self.player.pos.y < 0 or self.player.pos.y > self.height:
             self.lives = -1
             self.score += self.rewards["loss"]

        # Check Goal
        if pygame.sprite.spritecollideany(self.player, self.goal_group):
            self.score += self.rewards["win"]
            self.level += 1
            print(f"Level Completed! Advancing to Level {self.level}")
            if self.level > self.max_levels:
                self.level = 1
            self.init()

        self.player_group.draw(self.screen)
        self.block_group.draw(self.screen)
        self.trap_group.draw(self.screen)
        self.goal_group.draw(self.screen)

    def _handle_player_events(self):
        self.is_climbing = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == self.actions['up']:
                    self.is_climbing = True

    def game_over(self):
        return self.lives <= 0

    def getScore(self):
        return self.score

    def reset(self):
        self.level = 1
        self.init()
