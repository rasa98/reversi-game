import random
import time

import numpy as np
import pygame
import sys
import os
import platform

if __name__ == '__main__' and os.environ['USER'] != 'student':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
    sys.path.append(source_dir)
    print(f'cwd is : {os.getcwd()}')
    os.chdir('../')
from game_logic import Othello


def clear_console():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


class HumanPlayer:
    def __init__(self, name):
        self.name = name


human_player = HumanPlayer('You')


class OthelloGameGui:
    def __init__(self, min_turn_time=2, verbose=1):
        pygame.init()
        self.setup_display()

        self.min_turn_time = min_turn_time
        self.verbose = verbose

        self.players = None
        self.setup_game()

    def setup_game(self):
        self.game = Othello()
        self.ai = None
        self.last_played_move = None
        self.last_flipped_fields = set()

    def setup_display(self):
        """Set up the display window and font."""
        self.size = self.width, self.height = 800, 900  # 400, 500  # Increased height for space above the board
        self.rows, self.cols = 8, 8
        self.square_size = self.width // self.cols
        self.padding = 10  # 5
        self.disc_radius = (self.square_size - 2 * self.padding) // 2

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.gray = (128, 128, 128)
        self.green = (0, 128, 0)
        self.red = (220, 0, 0)
        self.lighter_green = (20, 198, 20)
        self.green_yellow = (154, 205, 50)

        # Set up the display
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Othello')
        self.font = pygame.font.Font(None, 90)

        self.label_font = pygame.font.Font(None, 35)  # Smaller font for labels

    def draw_board(self):
        """Draw the Othello board with space for labels."""
        top_space = 50  # Space for labels at the top
        for row in range(self.rows):
            for col in range(self.cols):
                field_color = self.green
                if (row, col) == self.last_played_move:
                    field_color = self.lighter_green
                # elif (row, col) in self.last_flipped_fields:
                #     field_color = self.green_yellow
                rect = pygame.Rect(col * self.square_size, top_space + row * self.square_size, self.square_size,
                                   self.square_size)
                pygame.draw.rect(self.screen, field_color, rect)
                pygame.draw.rect(self.screen, self.black, rect, 1)

    def draw_disc(self, row, col, color):
        """Draw a disc on the board."""
        top_space = 50
        x = col * self.square_size + self.square_size // 2
        y = top_space + row * self.square_size + self.square_size // 2
        pygame.draw.circle(self.screen, color, (x, y), self.disc_radius)

    def draw_diamond(self, row, col, color):
        """Draw a diamond (rotated square) centered in the board cell."""
        # Calculate the center of the cell
        top_space = 50
        center_x = col * self.square_size + self.square_size // 2
        center_y = top_space + row * self.square_size + self.square_size // 2

        # Calculate the size of the diamond relative to the cell
        diamond_size = self.square_size - 2 * self.padding
        half_diamond_size = diamond_size / 2

        # Define the points of the diamond centered at the calculated position
        points = [
            (center_x, center_y - half_diamond_size),  # Top point
            (center_x + half_diamond_size, center_y),  # Right point
            (center_x, center_y + half_diamond_size),  # Bottom point
            (center_x - half_diamond_size, center_y)  # Left point
        ]

        # Draw the diamond
        pygame.draw.polygon(self.screen, color, points)

    def draw_game_over(self):
        """Display the game-over screen."""
        top_space = 50
        self.update_display()
        # winner_label = 'you' if self.game.get_winner() == self.player_turn else 'ai'
        if self.game.get_winner() == 1:
            winner_label = self.players[0].name + ' Won!'
        elif self.game.get_winner() == 2:
            winner_label = self.players[1].name + ' Won!'
        else:
            winner_label = 'Its draw!'

        center = (self.width // 2, top_space + (self.height - 2 * top_space) // (8 / 3))
        label = f'{self.game.chips} {winner_label}'
        color = self.gray
        self.render_black_outline(self.font, center, label, color)

        label = 'press space to play again'
        center = (self.width // 2, top_space + (self.height - 2 * top_space) // 2)
        self.render_black_outline(self.font, center, label, color)

        pygame.display.flip()

        return self.wait_for_click_to_exit()

    def render_black_outline(self, font, center, label, color):
        # Render the outline
        text = font.render(label, True, self.black)
        for i in [-2, 0, 2]:
            for j in [-2, 0, 2]:
                if i != 0 or j != 0:
                    offset_center = (center[0] + i, center[1] + j)
                    text_rect = text.get_rect(center=offset_center)
                    self.screen.blit(text, text_rect)

        # Render the main text
        text = font.render(label, True, color)
        text_rect = text.get_rect(center=center)
        self.screen.blit(text, text_rect)

    def wait_for_click_to_exit(self):
        play_again = False
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return play_again
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    return play_again
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.setup_game()
                        clear_console()
                        play_again = True
                    return play_again
            clock.tick(10)  # Limit the loop to 10 iterations per second

    def update_display(self):
        """Update the display with the current game state."""
        top_space = 50
        self.screen.fill(self.green)
        self.draw_board()

        for row in range(self.rows):
            for col in range(self.cols):
                draw_shape_f = self.draw_disc
                if (row, col) in self.last_flipped_fields:
                    draw_shape_f = self.draw_diamond

                if self.game.board[row, col] == 1:
                    draw_shape_f(row, col, self.white)
                elif self.game.board[row, col] == 2:
                    draw_shape_f(row, col, self.black)

        is_1st_player_turn = self.game.player_turn == 1
        if is_1st_player_turn:
            highlight_color = self.white  # (144, 238, 144)
        else:
            highlight_color = self.black
        self.draw_valid_moves(highlight_color)

        # Draw score label
        w, b = self.game.chips
        text = f"({self.players[0].name}) White {w} - {b} Black ({self.players[1].name})"
        center = (self.width // 2, top_space // 2)
        self.render_text(text, center)

        text = f"Player turn: {self.ai.name}"
        center = (self.width // 2, self.height - top_space // 2)
        turn_text_color = self.white if is_1st_player_turn else self.black
        self.render_text(text, center, color=turn_text_color)

        pygame.display.flip()

    def render_text(self, text, center, color=(0, 0, 0)):
        score_label = self.label_font.render(text,
                                             True,
                                             color)
        score_label_rect = score_label.get_rect(center=center)
        self.screen.blit(score_label, score_label_rect)

    def draw_valid_moves(self, highlight_color):
        """Draw a border around valid move positions."""
        top_space = 50
        valid_moves = self.game.valid_moves()
        border_thickness = 5
        for move in valid_moves:
            row, col = move
            rect = pygame.Rect(
                col * self.square_size + border_thickness,
                top_space + row * self.square_size + border_thickness,
                self.square_size - 2 * border_thickness,
                self.square_size - 2 * border_thickness
            )
            pygame.draw.rect(self.screen, highlight_color, rect, border_thickness)

    def handle_click(self, pos):
        """Handle mouse clicks."""
        top_space = 50
        col = pos[0] // self.square_size
        row = (pos[1] - top_space) // self.square_size
        print(f"You clicked on cell ({int(row)}, {int(col)})\n")
        return row, col

    def main(self):
        """Main game loop."""
        assert self.players is not None, "You need to set configuration by calling set_match_conf method!"

        while not self.game.is_game_over():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.ai = self.players[self.game.player_turn - 1]
            self.update_display()
            field, fields_to_flip = self.make_move()

            self.last_played_move = field
            self.last_flipped_fields = fields_to_flip

        return self.draw_game_over()

    def make_move(self):
        if isinstance(self.ai, HumanPlayer):
            return self.make_human_move()
        return self.make_ai_move()

    def make_ai_move(self):
        start_time = time.perf_counter()
        field = self.play_ai_turn()
        end_time = time.perf_counter()
        ai_think_time = end_time - start_time
        if self.verbose:
            self.print_action_probabilities()
            print(f'{self.ai.name} chose {field}\n')
        # print(ai_think_time)
        delta_time = self.min_turn_time - ai_think_time
        if delta_time > 0:
            time.sleep(delta_time)
        fields_to_flip = self.game.valid_moves_to_reverse[field]
        self.game.play_move(field)
        return field, fields_to_flip

    def print_action_probabilities(self):
        if self.ai.action_probs is not None:
            array_2d = self.ai.action_probs.reshape(8, 8)
            array_2d_rounded = np.round(array_2d, 2)
            print(array_2d_rounded)
        if self.ai.estimated_value is not None:
            value = self.ai.estimated_value
            value_rounded = np.round(value, 2)
            print(f'win (value) estimate: {value_rounded}')

    def make_human_move(self):
        valid_moves = self.game.valid_moves()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.game._swap_player_turn()
                        self.game._calculate_next_valid_moves()
                        valid_moves = self.game.valid_moves()
                        self.update_display()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_field = self.handle_click(pygame.mouse.get_pos())
                    if clicked_field in valid_moves:
                        fields_to_flip = self.game.valid_moves_to_reverse[clicked_field]
                        self.game.play_move(clicked_field)
                        return clicked_field, fields_to_flip

    def play_ai_turn(self):
        fields, _ = self.ai.predict_best_move(self.game)
        return random.choice(fields)


def play_human_vs_human(verbose=1):
    game = OthelloGameGui(verbose=verbose)
    pl1 = HumanPlayer('white')
    pl2 = HumanPlayer('black')
    game.players = [pl1, pl2]

    # loop if playing again
    loop_game(game)


def play_human_vs_ai(ai_agent, human_turn=1, min_turn_time=2, verbose=1):
    game = OthelloGameGui(min_turn_time=min_turn_time, verbose=verbose)
    if human_turn == 1:
        game.players = [human_player, ai_agent]
    else:
        game.players = [ai_agent, human_player]

    # loop if playing again
    loop_game(game)


def play_ai_vs_ai(ai1, ai2, min_turn_time=2, verbose=1):
    game = OthelloGameGui(min_turn_time=min_turn_time, verbose=verbose)
    game.players = [ai1, ai2]

    # loop if playing again
    loop_game(game)


def loop_game(game):
    while game.main():
        game.setup_game()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    from read_all_agents import (alpha_200,
                                 alpha_30,
                                 best_ars,
                                 best_mlp_ppo,
                                 minmax_ga_best_depth_1,
                                 mcts_agent_500)

    play_ai_vs_ai(alpha_200, mcts_agent_500, min_turn_time=2, verbose=0)
