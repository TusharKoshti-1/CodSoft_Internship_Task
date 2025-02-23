# tic_tac_toe_ai.py
import tkinter as tk
from tkinter import messagebox, ttk
import math

class TicTacToeAI:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
        self.human_player = 'X'
        self.ai_player = 'O'

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind*3 : (row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

    def minimax(self, alpha, beta, maximizing_player):
        if self.current_winner:
            if self.current_winner == self.ai_player:
                return 1
            else:
                return -1
        elif not self.empty_squares():
            return 0

        if maximizing_player:
            max_eval = -math.inf
            for move in self.available_moves():
                self.make_move(move, self.ai_player)
                eval = self.minimax(alpha, beta, False)
                self.board[move] = ' '
                self.current_winner = None
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in self.available_moves():
                self.make_move(move, self.human_player)
                eval = self.minimax(alpha, beta, True)
                self.board[move] = ' '
                self.current_winner = None
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

class TicTacToeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unbeatable Tic-Tac-Toe AI")
        self.geometry("400x450")
        self.resizable(False, False)
        self.game = TicTacToeAI()
        self.buttons = []
        self.create_widgets()
        self.update_status("Your turn (X)")

    def create_widgets(self):
        # Game Board
        board_frame = ttk.Frame(self)
        board_frame.pack(pady=20)

        for i in range(9):
            row = i // 3
            col = i % 3
            btn = ttk.Button(
                board_frame,
                text=' ',
                width=8,
                command=lambda i=i: self.human_move(i),
                style='Game.TButton'
            )
            btn.grid(row=row, column=col, padx=5, pady=5)
            self.buttons.append(btn)

        # Status Label
        self.status_label = ttk.Label(self, text="", font=('Arial', 12))
        self.status_label.pack(pady=10)

        # Control Buttons
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10)

        ttk.Button(
            control_frame,
            text="New Game",
            command=self.new_game
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="Exit",
            command=self.destroy
        ).pack(side=tk.LEFT, padx=5)

        # Configure Styles
        self.style = ttk.Style()
        self.style.configure('Game.TButton', font=('Arial', 16, 'bold'))

    def update_status(self, message):
        self.status_label.config(text=message)

    def human_move(self, square):
        if self.game.board[square] == ' ' and not self.game.current_winner:
            self.buttons[square].config(text=self.game.human_player)
            self.game.make_move(square, self.game.human_player)
            
            if self.game.current_winner:
                self.end_game(f"Player {self.game.human_player} wins!")
            elif not self.game.empty_squares():
                self.end_game("It's a tie!")
            else:
                self.update_status("AI thinking...")
                self.after(500, self.ai_move)

    def ai_move(self):
        best_score = -math.inf
        best_move = None
        for move in self.game.available_moves():
            self.game.make_move(move, self.game.ai_player)
            score = self.game.minimax(-math.inf, math.inf, False)
            self.game.board[move] = ' '
            self.game.current_winner = None
            
            if score > best_score:
                best_score = score
                best_move = move

        self.buttons[best_move].config(text=self.game.ai_player)
        self.game.make_move(best_move, self.game.ai_player)
        
        if self.game.current_winner:
            self.end_game(f"AI {self.game.ai_player} wins!")
        elif not self.game.empty_squares():
            self.end_game("It's a tie!")
        else:
            self.update_status("Your turn (X)")

    def end_game(self, message):
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
        self.update_status(message)
        answer = messagebox.askyesno("Game Over", f"{message}\nPlay again?")
        if answer:
            self.new_game()
        else:
            self.destroy()

    def new_game(self):
        self.destroy()
        TicTacToeGUI().mainloop()

if __name__ == "__main__":
    app = TicTacToeGUI()
    app.mainloop()