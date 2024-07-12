import random
import time
import pygame
import sys
import os

if __name__ == '__main__' and os.environ['USER'] != 'student':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
    sys.path.append(source_dir)
    print(f'cwd is : {os.getcwd()}')
    os.chdir('../')
from game_logic import Othello
from read_all_agents import alpha_30


class OthelloGame:
    def __init__(self):
        pygame.init()
        self.game = Othello()
        self.setup_display()
        self.player_turn = None
        self.ai = None

    def set_match_conf(self, ai, player_turn):
        self.player_turn = player_turn
        self.ai = ai

    def setup_display(self):
        """Set up the display window and font."""
        self.size = self.width, self.height = 800, 800
        self.rows, self.cols = 8, 8
        self.square_size = self.width // self.cols
        self.padding = 10
        self.disc_radius = (self.square_size - 2 * self.padding) // 2

        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 128, 0)
        self.red = (255, 0, 0)

        # Set up the display
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Othello')
        self.font = pygame.font.Font(None, 55)

    def draw_board(self):
        """Draw the Othello board."""
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(col * self.square_size, row * self.square_size, self.square_size, self.square_size)
                pygame.draw.rect(self.screen, self.green, rect)
                pygame.draw.rect(self.screen, self.black, rect, 1)

    def draw_disc(self, row, col, color):
        """Draw a disc on the board."""
        x = col * self.square_size + self.square_size // 2
        y = row * self.square_size + self.square_size // 2
        pygame.draw.circle(self.screen, color, (x, y), self.disc_radius)

    def draw_game_over(self):
        """Display the game-over screen."""
        self.update_display()
        winner_label = 'you' if self.game.get_winner() == self.player_turn else 'ai'
        text = self.font.render(f'{self.game.chips}'
                                f' {winner_label}'
                                f' Won!',
                                True,
                                self.red)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
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
        self.screen.fill(self.green)
        self.draw_board()
        for row in range(self.rows):
            for col in range(self.cols):
                if self.game.board[row, col] == 1:
                    self.draw_disc(row, col, self.white)
                elif self.game.board[row, col] == 2:
                    self.draw_disc(row, col, self.black)

        if self.player_turn == self.game.player_turn:
            highlight_color = (144, 238, 144)
        else:
            highlight_color = self.red
        self.draw_valid_moves(highlight_color)
        pygame.display.flip()

    def draw_valid_moves(self, highlight_color):
        """Draw a light green border around valid move positions."""
        valid_moves = self.game.valid_moves()
        border_thickness = 3
        for move in valid_moves:
            row, col = move
            rect = pygame.Rect(
                col * self.square_size + border_thickness,
                row * self.square_size + border_thickness,
                self.square_size - 2 * border_thickness,
                self.square_size - 2 * border_thickness
            )
            pygame.draw.rect(self.screen, highlight_color, rect, border_thickness)

    def handle_click(self, pos):
        """Handle mouse clicks."""
        col = pos[0] // self.square_size
        row = pos[1] // self.square_size
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
                time.sleep(2 - (end_time - start_time))
                self.game.play_move(field)
            else:
                self.human_move()
        self.draw_game_over()

    def human_move(self):
        valid_moves = self.game.valid_moves()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_field = self.handle_click(pygame.mouse.get_pos())
                if clicked_field in valid_moves:
                    self.game.play_move(clicked_field)

    def play_ai_turn(self):
        fields, _ = self.ai.predict_best_move(self.game)
        return random.choice(fields)


if __name__ == "__main__":
    game = OthelloGame()
    game.set_match_conf(alpha_30, 2)
    game.main()
