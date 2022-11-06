import pygame
import math
class Utils:
    @staticmethod
    def blit_rotate_center(win, image, top_left, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(
            center=image.get_rect(topleft=top_left).center)
        win.blit(rotated_image, new_rect.topleft)
        
    @staticmethod
    def distanceBetweenTwoPoints(xPointA, yPointA, xPointB, yPointB):
        return math.sqrt((xPointA - xPointB)**2 + (yPointA - yPointB)**2)    
            
    