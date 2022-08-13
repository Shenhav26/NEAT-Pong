from pong import Game
import pygame
import time
import neat
import os
import pickle


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle

    def player_vs_ai(self, net):
        """
        Test the AI against a player by passing NEAT neural network
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            #slow time
            clock.tick(50)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            output = net.activate((self.right_paddle.y, abs(
                self.right_paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            if decision == 1:  # AI moves up
                self.game.move_paddle(left=False, up=True)
            elif decision == 2:  # AI moves down
                self.game.move_paddle(left=False, up=False)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pygame.K_DOWN]:
                self.game.move_paddle(left=True, up=False)

            self.game.draw(draw_score=True)
            pygame.display.update()

    def train_ai(self, genome1, genome2, config, draw=False):
        """
        Train the AI by passing two NEAT neural networks with our configuration
        object, each AI will play against eachother to determine their fitness.
        """
        run = True
        start_time = time.time()

        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        self.genome1 = genome1
        self.genome2 = genome2

        max_hits = 30

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

            game_info = self.game.loop()

            self.move_ai_paddles(net1, net2)

            if draw:
                self.game.draw(draw_score=False, draw_hits=True)

            pygame.display.update()

            duration = time.time() - start_time
            # stop calculate the fitness as soon one of the paddles missed the ball
            if game_info.left_score == 1 or game_info.right_score == 1 or game_info.left_hits >= max_hits:
                self.calculate_fitness(game_info, duration)
                break

        return False

    def move_ai_paddles(self, net1, net2):
        """
        Determine the movement of the right and left paddle based on the 
        neural network
        """
        players = [(self.genome1, net1, self.left_paddle, True), (self.genome2, net2, self.right_paddle, False)]
        for (genome, net, paddle, left) in players:
            output = net.activate(
                (paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
            decision = output.index(max(output))

            valid = True
            if decision == 0:  # stay on the same spot
                genome.fitness -= 0.01  # make it less possible to happend
            elif decision == 1:  # move up
                valid = self.game.move_paddle(left=left, up=True)
            else:  # move down
                valid = self.game.move_paddle(left=left, up=False)

            if not valid:  # if it's goes wide from the screen
                genome.fitness -= 1

    def calculate_fitness(self, game_info, duration):
        self.genome1.fitness += game_info.left_hits + duration
        self.genome2.fitness += game_info.right_hits + duration





def eval_genomes(genomes, config):
    """
    Run each genome with every other genome to give the most accurate fitness 
    to out AI (take more time to train), Can also run for just 2 pairs of genome
    to make it faster but with less accurate.
    """
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")

    for i, (genome_id1, genome1) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[min(i+1, len(genomes) - 1):]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            pong = PongGame(win, width, height)

            force_quit = pong.train_ai(genome1, genome2, config, draw=True)
            if force_quit:
                quit()



def run_neat(config):
    #load from a certain check point
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-#number')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #determine when to save a checkpoint after number of generations
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 30)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)




def test_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, width, height)
    pong.player_vs_ai(winner_net)

# load the configuration file
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # run NEAT with out configuration file
    run_neat(config)
    # playing vs our AI 
    test_network(config)
