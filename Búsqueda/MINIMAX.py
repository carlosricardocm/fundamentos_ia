import math
import copy

class TicTacToe:
    def __init__(self):
        # Estado inicial: tablero vacío (None representa casilla vacía)
        self.board = [[None, None, None],
                      [None, None, None],
                      [None, None, None]]
        self.current_player = 'X'  # X es MAX, O es MIN
    
    def S_o(self):
        """Retorna el estado inicial"""
        return [[None, None, None],
                [None, None, None],
                [None, None, None]]
    
    def PLAYER(self, state):
        """Regresa cual jugador mueve en el estado s"""
        x_count = sum(row.count('X') for row in state)
        o_count = sum(row.count('O') for row in state)
        
        # X siempre empieza, entonces si hay igual cantidad, es turno de X
        return 'X' if x_count == o_count else 'O'
    
    def ACTIONS(self, state):
        """Regresa cuales son acciones legales en el estado s"""
        actions = []
        for i in range(3):
            for j in range(3):
                if state[i][j] is None:
                    actions.append((i, j))
        return actions
    
    def RESULT(self, state, action):
        """Regresa el estado después de aplicar una acción a en un estado s"""
        new_state = copy.deepcopy(state)
        player = self.PLAYER(state)
        i, j = action
        new_state[i][j] = player
        return new_state
    
    def TERMINAL(self, state):
        """Verifica si un estado s es un estado terminal"""
        # Verificar si hay ganador
        if self.check_winner(state) is not None:
            return True
        
        # Verificar si el tablero está lleno (empate)
        for row in state:
            if None in row:
                return False
        return True
    
    def check_winner(self, state):
        """Verifica si hay un ganador y lo retorna ('X' o 'O'), o None si no hay"""
        # Verificar filas
        for row in state:
            if row[0] == row[1] == row[2] and row[0] is not None:
                return row[0]
        
        # Verificar columnas
        for col in range(3):
            if state[0][col] == state[1][col] == state[2][col] and state[0][col] is not None:
                return state[0][col]
        
        # Verificar diagonales
        if state[0][0] == state[1][1] == state[2][2] and state[0][0] is not None:
            return state[0][0]
        if state[0][2] == state[1][1] == state[2][0] and state[0][2] is not None:
            return state[0][2]
        
        return None
    
    def UTILITY(self, state):
        """Retorna la utilidad del estado terminal"""
        winner = self.check_winner(state)
        if winner == 'X':  # MAX gana
            return 1
        elif winner == 'O':  # MIN gana
            return -1
        else:  # Empate
            return 0
    
    def MAX_VALUE(self, state):
        """function MAX-VALUE(state):"""
        if self.TERMINAL(state):
            return self.UTILITY(state)
        
        v = -math.inf
        
        for action in self.ACTIONS(state):
            v = max(v, self.MIN_VALUE(self.RESULT(state, action)))
        
        return v
    
    def MIN_VALUE(self, state):
        """function MIN-VALUE(state):"""
        if self.TERMINAL(state):
            return self.UTILITY(state)
        
        v = math.inf
        
        for action in self.ACTIONS(state):
            v = min(v, self.MAX_VALUE(self.RESULT(state, action)))
        
        return v
    
    def MINIMAX_DECISION(self, state):
        """Retorna la mejor acción para el jugador actual"""
        player = self.PLAYER(state)
        best_action = None
        
        if player == 'X':  # MAX
            best_value = -math.inf
            for action in self.ACTIONS(state):
                value = self.MIN_VALUE(self.RESULT(state, action))
                if value > best_value:
                    best_value = value
                    best_action = action
        else:  # MIN
            best_value = math.inf
            for action in self.ACTIONS(state):
                value = self.MAX_VALUE(self.RESULT(state, action))
                if value < best_value:
                    best_value = value
                    best_action = action
        
        return best_action
    
    def print_board(self, state):
        """Imprime el tablero de forma visual"""
        print("\n")
        for i, row in enumerate(state):
            row_str = " | ".join([cell if cell is not None else " " for cell in row])
            print(f" {row_str} ")
            if i < 2:
                print("-----------")
        print("\n")
    
    def play_game(self, human_player='O'):
        """Juega una partida completa"""
        state = self.S_o()
        
        print("¡Bienvenido al Tic-Tac-Toe!")
        print(f"Tú eres '{human_player}' y la IA es '{'X' if human_player == 'O' else 'O'}'")
        
        while not self.TERMINAL(state):
            self.print_board(state)
            current = self.PLAYER(state)
            
            if current == human_player:
                # Turno del humano
                print("Tu turno:")
                valid = False
                while not valid:
                    try:
                        row = int(input("Fila (0-2): "))
                        col = int(input("Columna (0-2): "))
                        if (row, col) in self.ACTIONS(state):
                            action = (row, col)
                            valid = True
                        else:
                            print("Movimiento inválido. Intenta de nuevo.")
                    except:
                        print("Entrada inválida. Usa números 0-2.")
            else:
                # Turno de la IA
                print("Turno de la IA...")
                action = self.MINIMAX_DECISION(state)
                print(f"IA juega en: {action}")
            
            state = self.RESULT(state, action)
        
        # Juego terminado
        self.print_board(state)
        winner = self.check_winner(state)
        if winner == human_player:
            print("¡Ganaste! 🎉")
        elif winner is None:
            print("¡Empate! 🤝")
        else:
            print("La IA ganó. 🤖")


# Para jugar:
if __name__ == "__main__":
    game = TicTacToe()
    game.play_game(human_player='0')  # Puedes cambiar a 'X' si quieres empezar