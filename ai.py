import tkinter as tk
import numpy as np

# ------------------ OYUN SABITLERI ------------------
BOARD_SIZE = 7
EMPTY = 0
AI_PIECE = 1
HUMAN_PIECE = 2
MAX_MOVES = 50

def initialize_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    # Sol köşeler
    board[0, 0] = AI_PIECE
    board[0, 2] = AI_PIECE
    board[0, 4] = HUMAN_PIECE
    board[0, 6] = HUMAN_PIECE
    # Sağ köşeler
    board[6, 0] = HUMAN_PIECE
    board[6, 2] = HUMAN_PIECE
    board[6, 4] = AI_PIECE
    board[6, 6] = AI_PIECE
    return board

def valid_moves(board, piece):
    moves = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x, y] == piece:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if board[nx, ny] == 0:
                            moves.append(((x, y), (nx, ny)))
    return moves

def apply_move(board, move):
    (x1, y1), (x2, y2) = move
    board[x2, y2] = board[x1, y1]
    board[x1, y1] = EMPTY

def capture_pieces(board, move):
    """
    tahtayı mevcut hamleye göre günceller. Bu süreç, oyunun dinamik bir şekilde evrilmesine neden olur,
     çünkü her hamle tahtanın durumunu değiştirir.tahtanın güncellenmesi
    Taş yakalama kuralları:
    1) Line-based capture
    2) Pinned logic
    """
    (x1, y1), (x2, y2) = move
    current_piece = board[x2, y2]
    opponent_piece = HUMAN_PIECE if current_piece == AI_PIECE else AI_PIECE

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    captured_positions = []

    # 1) LINE-BASED CAPTURE
    for dx, dy in directions:
        temp_line = []
        step = 1
        while True:
            nx = x2 + step * dx
            ny = y2 + step * dy
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                captured_positions.extend(temp_line)
                break
            cell = board[nx, ny]
            if cell == opponent_piece:
                temp_line.append((nx, ny))
            elif cell == current_piece:
                captured_positions.extend(temp_line)
                break
            else:
                break
            step += 1

    # 2) PINNED LOGIC
    pinned = False
    for dx, dy in directions:
        nx = x2 + dx
        ny = y2 + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if board[nx, ny] == opponent_piece:
                ox = x2 - dx
                oy = y2 - dy
                if not (0 <= ox < BOARD_SIZE and 0 <= oy < BOARD_SIZE):
                    pinned = True
                    break
                if board[ox, oy] == opponent_piece:
                    pinned = True
                    break

    if pinned:
        captured_positions.append((x2, y2))

    for cx, cy in captured_positions:
        board[cx, cy] = EMPTY

    return len(captured_positions)

def check_game_end(board, move_count):
    ai_count = np.count_nonzero(board == AI_PIECE)
    human_count = np.count_nonzero(board == HUMAN_PIECE)
    if ai_count == 0:
        return True, "Human Wins!"
    if human_count == 0:
        return True, "AI Wins!"
    if move_count >= MAX_MOVES:
        return True, "Draw!"
    return False, None

# ------------------ MINIMAX EKLEMELERİ Bir oyun ağacında tüm olası hamleleri analiz ederek, her oyuncunun en iyi hareketi yapacağını varsayar.------------------
def evaluate_board(board):
    """
    Tahtadaki mevcut durumun değerini (skorunu) döndürmektir
    Basit bir değerlendirme:
    +1 puan => AI_PIECE
    -1 puan => HUMAN_PIECE
    
    """
    ai_count = np.count_nonzero(board == AI_PIECE) #Tahtadaki yapay zekâ taşlarının sayısını belirler.

    human_count = np.count_nonzero(board == HUMAN_PIECE)  #Tahtadaki insan taşlarının sayısını belirler.
    return ai_count - human_count  # Daha çok taşı olan kazanır, Temel görevi; tahtadaki taşların sayısına bakarak hangi tarafın daha avantajlı olduğunu analiz etmektir.

def minimax(board, depth, alpha, beta, maximizingPlayer):
    """
    Minimax + Alpha-Beta pruning
    maximizingPlayer = True => AI'nin sırası
    maximizingPlayer = False => Human'ın sırası
    """
    # Oyun bitmiş mi veya derinlik sıfır mı?algoritmanın bir oyun ağacında kaç adım öteye kadar ilerleyeceğini belirtir. 
    # Bu kavramın amacı, oyun sırasında tüm olası hamlelerin incelenmesini sınırlamak ve bu sayede performansı artırmaktır.
    # Algoritmanın ne kadar ileriye bakarak strateji geliştireceğini belirler.Performans ve doğruluk arasında denge kurar.
    if depth == 0:
        return evaluate_board(board)

    # AI moves
    if maximizingPlayer:
        bestVal = float('-inf')
        moves = valid_moves(board, AI_PIECE)
        if not moves:  # Hamle yoksa
            return evaluate_board(board)
        for move in moves:
            # Tahtayı kopyala, AI'nin hamleleri değerlendirilirken pruning uygulanır
            board_copy = np.copy(board)
            apply_move(board_copy, move) 
            capture_pieces(board_copy, move)
            value = minimax(board_copy, depth - 1, alpha, beta, False)
            bestVal = max(bestVal, value)
            alpha = max(alpha, bestVal) #Alpha (AI'nin garantili en iyi skoru):
            if beta <= alpha:   #Kesme Mantığı (Pruning)
                break
        return bestVal

    # Human moves
    else:
        bestVal = float('inf')
        moves = valid_moves(board, HUMAN_PIECE)
        if not moves:
            return evaluate_board(board)
        for move in moves: #İnsan oyuncunun hamleleri değerlendirilirken de pruning yapılır
            board_copy = np.copy(board)
            apply_move(board_copy, move)
            capture_pieces(board_copy, move)
            value = minimax(board_copy, depth - 1, alpha, beta, True)
            bestVal = min(bestVal, value)  
            beta = min(beta, bestVal) #Beta (İnsan oyuncunun garantili en kötü skoru)
            if beta <= alpha:
                break
        return bestVal

def get_best_move_minimax(board, depth=6):
    """
    Tüm geçerli hamleleri dener, minimax fonksiyonunu çalıştırır
    ve en iyi sonucu veren hamleyi döndürür.
    """
    best_score = float('-inf')
    best_move = None
    #Tüm geçerli hamleleri hesaplar
    all_moves = valid_moves(board, AI_PIECE)
    if not all_moves:
        return None

    for move in all_moves:
        temp_board = np.copy(board)
        apply_move(temp_board, move)
        capture_pieces(temp_board, move)
        #Her hamle için tahtayı kopyalar ve uygular
        score = minimax(temp_board, depth - 1, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=False)
        #Hamle sonrası Minimax algoritmasını çalıştırır
        if score > best_score:
            best_score = score
            best_move = move #En yüksek skoru veren hamleyi döndürür
    return best_move
# ------------------ MINIMAX EKLEMELERİ SONU ------------------


# ------------------ TKINTER GUI SINIFI ------------------
class OrtaOyunuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Orta Oyunu - GUI")

        # Başlangıçta tahta
        self.board = initialize_board()
        self.move_count = 0

        # AI başlasın
        self.turn = AI_PIECE
        self.selected_piece = None
        self.game_over = False

        # 2 hamle kuralı
        self.moves_left_for_turn = 2
        self.last_moved_pieces = set()

        self.ai_kills = 0
        self.human_kills = 0

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        self.buttons = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        self.info_label = tk.Label(self.main_frame, text="AI'nın sırası.")
        self.info_label.grid(row=0, column=0, columnspan=BOARD_SIZE, pady=5)

        self.kills_label = tk.Label(self.main_frame, text="AI Kills: 0 | Human Kills: 0")
        self.kills_label.grid(row=1, column=0, columnspan=BOARD_SIZE, pady=5)

        self.create_board_buttons()
        self.update_board_display()

        # AI'nın ilk turunu .5 sn sonra başlat
        self.root.after(500, self.ai_turn_if_needed)

    def create_board_buttons(self):
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                btn = tk.Button(
                    self.main_frame,
                    width=4, height=2,
                    command=lambda r=x, c=y: self.on_cell_click(r, c)
                )
                btn.grid(row=y+2, column=x, padx=2, pady=2)
                self.buttons[x][y] = btn

    def update_board_display(self):
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                piece = self.board[x, y]
                if piece == EMPTY:
                    self.buttons[x][y].config(text="", bg="white")
                elif piece == AI_PIECE:
                    self.buttons[x][y].config(text="▲", fg="red", bg="white")
                elif piece == HUMAN_PIECE:
                    self.buttons[x][y].config(text="●", fg="blue", bg="white")

        self.kills_label.config(
            text=f"AI Kills: {self.ai_kills} | Human Kills: {self.human_kills}"
        )

    def on_cell_click(self, x, y):
        """İnsan oyuncu için 2 hamle hakkı; aynı taşı 2 kez oynayamaz."""
        if self.game_over or self.turn != HUMAN_PIECE or self.moves_left_for_turn <= 0:
            return

        # İnsan tek taşı kaldıysa 1 hamle
        human_count = np.count_nonzero(self.board == HUMAN_PIECE)
        if human_count == 1:
            self.moves_left_for_turn = 1

        if self.selected_piece is None:
            # Taş seçme
            if (self.board[x, y] == HUMAN_PIECE) and ((x, y) not in self.last_moved_pieces):
                self.selected_piece = (x, y)
                self.info_label.config(
                    text=f"Seçilen taş: ({x},{y}). Hedef kareye tıklayın."
                )
            else:
                self.info_label.config(
                    text="Bu turda oynanmamış kendi taşınızı seçmelisiniz!"
                )
        else:
            # Hamle
            from_x, from_y = self.selected_piece
            to_x, to_y = x, y

            if self.is_valid_move(from_x, from_y, to_x, to_y, HUMAN_PIECE):
                self.perform_move((from_x, from_y), (to_x, to_y))
                self.last_moved_pieces.add((to_x, to_y))
                self.selected_piece = None
                self.moves_left_for_turn -= 1

                if self.moves_left_for_turn == 0:
                    # Sıra AI'ya
                    self.turn = AI_PIECE
                    self.moves_left_for_turn = 2
                    self.last_moved_pieces.clear()
                    self.info_label.config(text="AI'nın sırası.")
                    self.root.after(500, self.ai_turn_if_needed)
                else:
                    self.info_label.config(text="İkinci hamlenizi yapabilirsiniz (farklı bir taş).")
            else:
                self.info_label.config(text="Geçersiz hamle. Tekrar deneyin.")
                self.selected_piece = None

    def is_valid_move(self, from_x, from_y, to_x, to_y, piece):
        if self.board[from_x, from_y] != piece:
            return False
        if not (0 <= to_x < BOARD_SIZE and 0 <= to_y < BOARD_SIZE):
            return False
        if self.board[to_x, to_y] != EMPTY:
            return False
        # Yatay/dikey 1 kare gitmeli
        if abs(from_x - to_x) + abs(from_y - to_y) != 1:
            return False
        return True

    def perform_move(self, source, destination):
        apply_move(self.board, (source, destination))
        captured_count = capture_pieces(self.board, (source, destination))
        self.move_count += 1

        moved_piece = self.board[destination[0], destination[1]]
        if moved_piece == HUMAN_PIECE:
            self.human_kills += captured_count
        else:
            self.ai_kills += captured_count

        self.update_board_display()

        ended, result = check_game_end(self.board, self.move_count)
        if ended:
            self.game_over = True
            self.info_label.config(text=result)

    # ----------------------------------------------------------------
    # YENİDEN DÜZENLENMİŞ AI TURU:
    #   "2 farklı taş" zorunluluğu + (Minimax kullanımı)
    # ----------------------------------------------------------------
    def ai_turn_if_needed(self):
        """
        AI, elinde 1 taş kalmadığı sürece her zaman 2 farklı taşı oynar.
        1) Eğer AI'nin sadece 1 taşı varsa => tek hamle yapar.
        2) Eğer AI'nin 2 veya daha fazla taşı varsa => arka arkaya 2 hamle,
           üstelik 2 FARKLI taş ile hamle yapar.
        """
        if self.game_over or self.turn != AI_PIECE:
            return

        ai_count = np.count_nonzero(self.board == AI_PIECE)

        if ai_count <= 1:
            # Tek hamle
            self.make_ai_move_for_one_piece()
            self.end_ai_turn()
            return

        # 2 hamle
        self.last_moved_pieces.clear()

        # 1. hamle
        first_move_done = self.make_ai_move_for_new_piece()
        if not first_move_done:
            self.end_ai_turn()
            return

        # 2. hamle
        second_move_done = self.make_ai_move_for_new_piece()
        self.end_ai_turn()

    # ------------------ AI'nin tek taşı varsa (minimax'siz) ------------------
    def make_ai_move_for_one_piece(self):
        all_moves = valid_moves(self.board, AI_PIECE)
        if not all_moves:
            return
        # Derinliği 6 olarak ayarlıyoruz
        move = get_best_move_minimax(self.board, depth=6)
        if move is None:
            return
        self.perform_move(move[0], move[1])

    # ------------------ AI'nin 2 veya daha fazla taşı varsa ------------------
    def make_ai_move_for_new_piece(self):
        """
        2 veya daha fazla taşı varsa, henüz bu turda oynanmamış bir taştan
        hamle bulup uygular (minimax). Bulamazsa False döner.
        """
        # Tüm hamleleri al
        all_moves = valid_moves(self.board, AI_PIECE)
        # Bu turda kullanılmamış taşları filtrele
        possible_moves = []
        for (sx, sy), (dx, dy) in all_moves:
            if (sx, sy) not in self.last_moved_pieces:
                possible_moves.append(((sx, sy), (dx, dy)))

        if not possible_moves:
            return False

        # Derinliği 6 olarak ayarlıyoruz
        best_move = get_best_move_minimax(self.board, depth=6)
        if (best_move is None) or (best_move not in possible_moves):
            # Minimax bir hamle bulsa da, o hamle bu turda kullanılmamış taştan değilse
            if possible_moves:
                best_move = possible_moves[0]
            else:
                return False

        self.perform_move(best_move[0], best_move[1])
        self.last_moved_pieces.add((best_move[1][0], best_move[1][1]))
        return True

    def end_ai_turn(self):
        self.last_moved_pieces.clear()
        self.moves_left_for_turn = 2
        self.turn = HUMAN_PIECE
        self.info_label.config(text="Sıra sizde.")

# ------------------ UYGULAMA BASLATMA ------------------
if __name__ == "__main__":
    root = tk.Tk()
    game_gui = OrtaOyunuGUI(root)
    root.mainloop()
