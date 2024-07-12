import random
import time

import numpy as np
import pygame
import sys
import os

if __name__ == '__main__' and os.environ['USER'] != 'student':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
    sys.path.append(source_dir)
    print(f'cwd is : {os.getcwd()}')
    os.chdir('../')
from game_logic import Othello
from read_all_agents import alpha_200


class OthelloGameGui:
    def __init__(self):
        pygame.init()
        self.game = Othello()
        self.setup_display()
        self.player_turn = None
        self.ai = None

        self.last_played_move = None
        self.last_flipped_fields = set()

    def set_match_conf(self, ai, player_turn):
        self.player_turn = player_turn
        self.ai = ai

    def setup_display(self):
        """Set up the display window and font."""
        self.size = self.width, self.height = 800, 850  # Increased height for space above the board
        self.rows, self.cols = 8, 8
        self.square_size = self.width // self.cols
        self.padding = 10
        self.disc_radius = (self.square_size - 2 * self.padding) // 2

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 128, 0)
        self.red = (255, 0, 0)
        self.lighter_green = (20, 198, 20)
        self.green_yellow = (154, 205, 50)

        # Set up the display
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Othello')
        self.font = pygame.font.Font(None, 95)
        self.label_font = pygame.font.Font(None, 50)  # Smaller font for labels

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
        winner_label = 'you' if self.game.get_winner() == self.player_turn else 'ai'
        text = self.font.render(f'{self.game.chips} {winner_label} Won!', True, self.red)
        text_rect = text.get_rect(center=(self.width // 2, top_space + (self.height - top_space) // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

        self.wait_for_click_to_exit()

    @staticmethod
    def wait_for_click_to_exit():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                    return

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

        if self.player_turn == self.game.player_turn:
            highlight_color = (144, 238, 144)
        else:
            highlight_color = self.red
        self.draw_valid_moves(highlight_color)

        # Draw score label
        w, b = self.game.chips

        if self.player_turn == 1:
            text = f"Score: White (human) {w} - Black {b}"
        else:
            text = f"Score: White {w} - Black (human) {b}"
        score_label = self.label_font.render(text,
                                             True,
                                             self.black)
        score_label_rect = score_label.get_rect(center=(self.width // 2, top_space // 2))
        self.screen.blit(score_label, score_label_rect)

        pygame.display.flip()

    def draw_valid_moves(self, highlight_color):
        """Draw a border around valid move positions."""
        top_space = 50
        valid_moves = self.game.valid_moves()
        border_thickness = 3
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
        print(f"Clicked on cell ({row}, {col})")
        return row, col

    def main(self):
        """Main game loop."""
        assert self.ai is not None, "You need to select ai agent!"

        while not self.game.is_game_over():
            self.update_display()
            if self.player_turn != self.game.player_turn:
                start_time = time.perf_counter()
                field = self.play_ai_turn()
                end_time = time.perf_counter()
                ai_think_time = end_time - start_time

                try:
                    array_2d = self.ai.action_probs.reshape(8, 8)
                    array_2d_rounded = np.round(array_2d, 2)
                    print(array_2d_rounded)
                except AttributeError as e:
                    print(f"Save action_probs for this agent!")

                # print(ai_think_time)
                if 2 - ai_think_time > 0:
                    time.sleep(2 - ai_think_time)
                fields_to_flip = self.game.valid_moves_to_reverse[field]
                self.game.play_move(field)
            else:
                field, fields_to_flip = self.human_move()
            self.last_played_move = field
            self.last_flipped_fields = fields_to_flip

        self.draw_game_over()

    def human_move(self):
        valid_moves = self.game.valid_moves()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_field = self.handle_click(pygame.mouse.get_pos())
                    if clicked_field in valid_moves:
                        fields_to_flip = self.game.valid_moves_to_reverse[clicked_field]
                        self.game.play_move(clicked_field)
                        return clicked_field, fields_to_flip

    def play_ai_turn(self):
        fields, _ = self.ai.predict_best_move(self.game)
        return random.choice(fields)


if __name__ == "__main__":
    game = OthelloGameGui()
    game.set_match_conf(alpha_200, 1)
    game.main()
