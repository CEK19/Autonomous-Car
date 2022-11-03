class Environment:
    def __init__(self, player) -> None:
        self.player = player
        self.x = player.x
        self.y = player.y
        self.cast = player.cast
    
    def update(self):
        self.player.move()
        self.x = self.player.x
        self.y = self.player.y
        self.cast = self.player.cast
        
class Player:
    def __init__(self) -> None:
        self.x = 0
        self.y = 1
        self.cast = []
        
    def move(self):
        self.x = self.x + 1
        self.y = self.y + 1
        self.cast = [1, 2]
        
player = Player()
print(player.x, player.y)
env = Environment(player)
print(player.x, player.y, player.cast , env.x, env.y, env.cast)
env.update()
print(player.x, player.y, player.cast , env.x, env.y, env.cast)