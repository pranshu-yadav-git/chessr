import cv2
import numpy as np
import chess
import chess.engine
import time
import threading
import os 
import requests
import zipfile
import io
import platform
from scipy.spatial import distance

class CameraChessAnalyzer:
    def __init__(self, camera_id=0):
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")
            
        # Chess engine setup
        self.engine_path = self.download_stockfish()
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        
        # Chess board state
        self.board = chess.Board()
        self.corners = None
        self.square_centers = None
        self.piece_positions = {}
        self.best_move = None
        self.best_score = None
        
        # Analysis flags
        self.running = True
        self.board_detected = False
        
        # Start the processing threads
        self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
        self.analysis_thread.start()
    
    def download_stockfish(self):
        """Download Stockfish chess engine if not present"""
        stockfish_dir = "stockfish"
        os.makedirs(stockfish_dir, exist_ok=True)
        
        system = platform.system()
        if system == "Windows":
            stockfish_path = os.path.join(stockfish_dir, "stockfish.exe")
            url = "https://stockfishchess.org/files/stockfish_15.1_win_x64_avx2.zip"
        else:  # Linux/Mac
            stockfish_path = os.path.join(stockfish_dir, "stockfish")
            url = "https://stockfishchess.org/files/stockfish-15.1-linux.zip"
        
        # Check if Stockfish already exists
        if os.path.exists(stockfish_path):
            return stockfish_path
            
        print("Downloading Stockfish chess engine...")
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(stockfish_dir)
        
        # Find the executable
        for root, dirs, files in os.walk(stockfish_dir):
            for file in files:
                if file.lower().startswith("stockfish") and (".exe" in file.lower() or not "." in file):
                    path = os.path.join(root, file)
                    # Make executable on Linux/Mac
                    if system != "Windows":
                        os.chmod(path, 0o755)
                    return path
        
        raise Exception("Could not find Stockfish executable after download")
    
    def detect_chessboard(self, frame): 
        """Detect chessboard in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to find chessboard corners (8x8 internal corners)
        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Calculate the complete 8x8 grid
            # We need to extrapolate the outer corners
            corners = corners.reshape(-1, 2)
            
            # Sort corners by row and column
            rows = []
            for i in range(7):
                rows.append(corners[i*7:(i+1)*7])
            
            # Calculate vectors for grid extrapolation
            h_vector = (rows[0][1] - rows[0][0]) / 7  # Horizontal unit vector
            v_vector = (rows[1][0] - rows[0][0]) / 7  # Vertical unit vector
            
            # Create complete 9x9 grid (including outer points)
            complete_grid = []
            for i in range(9):
                row = []
                for j in range(9):
                    # Calculate position with extrapolation for outer points
                    pos = rows[0][0] - h_vector - v_vector + i * v_vector * 7 + j * h_vector * 7
                    row.append(pos)
                complete_grid.append(row)
            
            self.corners = np.array(complete_grid, dtype=np.float32)
            
            # Calculate centers of the 64 squares
            square_centers = []
            for i in range(8):
                for j in range(8):
                    # Calculate center of square from its 4 corners
                    tl = complete_grid[i][j]
                    tr = complete_grid[i][j+1]
                    bl = complete_grid[i+1][j]
                    br = complete_grid[i+1][j+1]
                    center = (tl + tr + bl + br) / 4
                    square_centers.append((center[0], center[1]))
            
            self.square_centers = square_centers
            self.board_detected = True
            return True
        
        return False
    
    def get_piece_at_square(self, frame, square_idx):
        """Determine if there's a piece at the given square and what color it is"""
        if not self.square_centers or square_idx >= len(self.square_centers):
            return None
            
        center = self.square_centers[square_idx]
        center = (int(center[0]), int(center[1]))
        
        # Extract a small region around the center
        size = 15  # Size of the region to analyze
        roi = frame[max(0, center[1]-size):min(frame.shape[0], center[1]+size), 
                   max(0, center[0]-size):min(frame.shape[1], center[0]+size)]
        
        if roi.size == 0:
            return None
            
        # Calculate color features
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check if there's a piece by looking at variance
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray_roi)
        
        if variance < 100:  # Empty square has low variance
            return None
        
        # Determine piece color (simplified approach)
        average_v = np.mean(hsv_roi[:,:,2])  # Value channel average
        if average_v < 100:
            return 'b'  # Black piece
        else:
            return 'w'  # White piece
    
    def detect_pieces(self, frame):
        """Detect chess pieces on the board"""
        if not self.square_centers:
            return
            
        new_positions = {}
        for i in range(64):
            piece_color = self.get_piece_at_square(frame, i)
            if piece_color:
                new_positions[i] = piece_color
        
        self.piece_positions = new_positions
        
        # Convert detected positions to FEN
        self.board = self.positions_to_board(new_positions)
    
    def positions_to_board(self, positions):
        """Convert detected piece positions to a chess.Board object"""
        # Initialize empty board
        board = chess.Board(fen="8/8/8/8/8/8/8/8 w KQkq - 0 1")
        
        # Place pieces based on detected positions
        for square_idx, color in positions.items():
            # Determine piece type (simplified - we're only detecting color)
            # In a real implementation, you'd use a CNN to classify piece types
            # For simplicity, we'll place pawns for all pieces
            piece_type = chess.PAWN
            
            # Create piece and place it on the board
            piece = chess.Piece(piece_type, color == 'w')
            board.set_piece_at(square_idx, piece)
        
        # Assuming white to move (in a real implementation, you'd detect this)
        board.turn = chess.WHITE
        
        # Assume no castling rights (could be improved)
        board.castling_rights = chess.BB_EMPTY
        
        return board
    
    def find_best_move(self):
        """Find the best move for the current position"""
        try:
            if self.board.is_valid():
                # Use the chess engine to find the best move
                result = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
                best_move = result["pv"][0] if "pv" in result and result["pv"] else None
                score = result["score"].relative.score(mate_score=10000) if "score" in result else None
                return best_move, score
        except Exception as e:
            print(f"Engine analysis error: {e}")
        
        return None, None
    
    def determine_player_color(self):
        """Determine which color the player is using based on board orientation and pieces"""
        if not self.board_detected or not self.corners.any():
            return None
        
        # Method 1: Check board orientation
        # In standard chess setups, white pieces are on ranks 1-2 (bottom)
        # Black pieces are on ranks 7-8 (top)
        # We can check if the board is oriented with white at bottom or top
        
        # Get the direction from camera's perspective
        # If the 0,0 corner (top-left in image) corresponds to a8 (top-left in standard notation)
        # then the player is likely playing as black (bottom of image)
        # If the 0,0 corner corresponds to h1, then player is likely white
        
        # Algorithm: Check color concentration on different sides of the board
        if len(self.piece_positions) < 10:  # Need enough pieces to make a determination
            return None
        
        # Count pieces in the "near half" and "far half" of the board
        near_half_white = 0
        near_half_black = 0
        far_half_white = 0
        far_half_black = 0
        
        for square_idx, color in self.piece_positions.items():
            rank = square_idx // 8  # 0-7, where 0 is the top rank from white's perspective
            
            if rank < 4:  # Far half (ranks 5-8)
                if color == 'w':
                    far_half_white += 1
                else:
                    far_half_black += 1
            else:  # Near half (ranks 1-4)
                if color == 'w':
                    near_half_white += 1
                else:
                    near_half_black += 1
        
        # If more white pieces are in the near half, player is likely black
        # If more black pieces are in the near half, player is likely white
        if near_half_white > far_half_white and far_half_black > near_half_black:
            return chess.BLACK  # Player is black
        elif near_half_black > far_half_black and far_half_white > near_half_white:
            return chess.WHITE  # Player is white
        
        # If the above is inconclusive, try another approach
        # Check for kings and queens (usually good indicators)
        # This would require piece type recognition
        
        # For now, return a default if we couldn't determine
        return None

    def get_player_perspective_move(self):
        """Get a move recommendation from the player's perspective"""
        player_color = self.determine_player_color()
        
        if player_color is None:
            # If we can't determine player color, just return the best move overall
            return self.best_move, self.best_score
        
        # If player is white, recommend best move for white
        # If player is black, recommend best move for black
        if player_color == chess.WHITE:
            if self.board.turn == chess.WHITE:
                # Player's turn - recommend move for white
                return self.best_move, self.best_score
            else:
                # Opponent's turn - no recommendation
                return None, None
        else:  # Player is black
            if self.board.turn == chess.BLACK:
                # Player's turn - recommend move for black
                return self.best_move, self.best_score
            else:
                # Opponent's turn - no recommendation
                return None, None
    
    def draw_board_outline(self, frame):
        """Draw the detected chessboard outline"""
        if not self.corners.any():
            return frame
            
        # Draw outer chessboard border
        for i in range(8):
            # Top edge
            cv2.line(frame, 
                   (int(self.corners[0][i][0]), int(self.corners[0][i][1])), 
                   (int(self.corners[0][i+1][0]), int(self.corners[0][i+1][1])), 
                   (0, 0, 255), 2)
            # Bottom edge
            cv2.line(frame, 
                   (int(self.corners[8][i][0]), int(self.corners[8][i][1])), 
                   (int(self.corners[8][i+1][0]), int(self.corners[8][i+1][1])), 
                   (0, 0, 255), 2)
            # Left edge
            cv2.line(frame, 
                   (int(self.corners[i][0][0]), int(self.corners[i][0][1])), 
                   (int(self.corners[i+1][0][0]), int(self.corners[i+1][0][1])), 
                   (0, 0, 255), 2)
            # Right edge
            cv2.line(frame, 
                   (int(self.corners[i][8][0]), int(self.corners[i][8][1])), 
                   (int(self.corners[i+1][8][0]), int(self.corners[i+1][8][1])), 
                   (0, 0, 255), 2)
                   
        # Draw square centers
        if self.square_centers:
            for center in self.square_centers:
                cv2.circle(frame, (int(center[0]), int(center[1])), 3, (255, 0, 0), -1)
                
        return frame
    
    def draw_best_move(self, frame):
        """Draw the best move as an overlay on the frame from player's perspective"""
        # Get move from player's perspective
        move, score = self.get_player_perspective_move()
        
        if not move or not self.square_centers:
            return frame
        
        # Get the source and destination squares
        from_square = move.from_square
        to_square = move.to_square
        
        # Get pixel coordinates
        from_pos = self.square_centers[from_square]
        to_pos = self.square_centers[to_square]
        
        # Draw the move as an arrow
        overlay = frame.copy()
        cv2.arrowedLine(overlay, 
                      (int(from_pos[0]), int(from_pos[1])), 
                      (int(to_pos[0]), int(to_pos[1])), 
                      (0, 255, 0), 3, tipLength=0.3)
        
        # Add score text if available
        if score is not None:
            score_text = f"+{score/100:.2f}" if score >= 0 else f"{score/100:.2f}"
            move_text = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
            
            # Add player color to the display
            player_color = self.determine_player_color()
            color_text = "White" if player_color == chess.WHITE else "Black" if player_color == chess.BLACK else "Unknown"
            
            cv2.putText(overlay, f"You: {color_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Best: {move_text} ({score_text})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Apply overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def analysis_loop(self):
        """Run chess analysis in a background thread"""
        while self.running:
            if self.board_detected:
                # Find best move for current position
                self.best_move, self.best_score = self.find_best_move()
            time.sleep(0.5)  # Update analysis every half second
    
    def run(self):
        """Main processing loop with player color selection"""
        try:
            last_detection_time = 0
            manually_selected_color = None
            
            print("Press 'w' to indicate you're playing as White")
            print("Press 'b' to indicate you're playing as Black")
            
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Try to detect chessboard periodically
                current_time = time.time()
                if not self.board_detected or current_time - last_detection_time > 10:
                    if self.detect_chessboard(frame):
                        print("Chessboard detected")
                        last_detection_time = current_time
                
                # If board detected, detect pieces and draw overlays
                if self.board_detected:
                    self.detect_pieces(frame)
                    frame = self.draw_board_outline(frame)
                    frame = self.draw_best_move(frame)
                    
                    # Display the player color
                    player_color = manually_selected_color or self.determine_player_color()
                    color_str = "White" if player_color == chess.WHITE else "Black" if player_color == chess.BLACK else "Detecting..."
                    cv2.putText(frame, f"You are playing as: {color_str}", 
                              (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 0), 2)
                    
                    # Instructions for manual selection
                    cv2.putText(frame, "Press 'w' for White, 'b' for Black", 
                              (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 200, 255), 1)
                else:
                    # Prompt to position camera correctly
                    cv2.putText(frame, "Position camera to view the entire chessboard", 
                              (50, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow("Chess Analyzer", frame)
                
                # Handle key presses
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
                elif key == ord('w') or key == ord('W'):
                    manually_selected_color = chess.WHITE
                    print("You selected: Playing as White")
                elif key == ord('b') or key == ord('B'):
                    manually_selected_color = chess.BLACK
                    print("You selected: Playing as Black")
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        if self.engine:
            self.engine.quit()
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

# Advanced version with piece type recognition using machine learning
class AdvancedChessAnalyzer(CameraChessAnalyzer):
    def __init__(self, camera_id=0):
        # Initialize parent class
        super().__init__(camera_id)
        
        # Load piece recognition model
        # In a full implementation, this would be a trained CNN model
        self.piece_classifier = self.load_piece_classifier()
        
    def load_piece_classifier(self):
        """
        In a real implementation, this would load a trained model for chess piece classification
        For this example, we'll use a placeholder function
        """
        # This is where you'd load your trained model, e.g.:
        # return tf.keras.models.load_model('chess_piece_classifier.h5')
        print("Note: Using simplified piece detection. For best results, implement a CNN model.")
        return None
        
    def get_piece_at_square(self, frame, square_idx):
        """Enhanced version that detects piece type as well as color"""
        piece_color = super().get_piece_at_square(frame, square_idx)
        
        if not piece_color:
            return None
            
        # In a real implementation, you would:
        # 1. Extract the piece image
        # 2. Preprocess it for your model
        # 3. Use the model to classify the piece type
        # 4. Return a chess.Piece object
        
        # For this example, we'll just detect pawns for simplicity
        return piece_color
        
    def positions_to_board(self, positions):
        """Improved version that sets piece types correctly"""
        # Start with empty board
        board = chess.Board(fen="8/8/8/8/8/8/8/8 w KQkq - 0 1")
        
        # Place detected pieces
        for square_idx, color in positions.items():
            is_white = color == 'w'
            
            # In a real implementation, you would determine the piece type
            # based on your classifier output
            # For now we'll use a simplified approach based on initial positions
            rank = square_idx // 8
            file = square_idx % 8
            
            # Use initial board setup as a heuristic
            if rank == 0 or rank == 7:  # Back ranks
                if file == 0 or file == 7:
                    piece_type = chess.ROOK
                elif file == 1 or file == 6:
                    piece_type = chess.KNIGHT
                elif file == 2 or file == 5:
                    piece_type = chess.BISHOP
                elif file == 3:
                    piece_type = chess.QUEEN
                elif file == 4:
                    piece_type = chess.KING
                else:
                    piece_type = chess.PAWN
            elif rank == 1 or rank == 6:  # Pawn ranks
                piece_type = chess.PAWN
            else:
                # For other squares, default to pawn
                piece_type = chess.PAWN
            
            piece = chess.Piece(piece_type, is_white)
            board.set_piece_at(square_idx, piece)
        
        # Try to determine whose turn it is
        # In a real implementation, you might look at recent moves or
        # use a UI element for the user to indicate this
        white_count = sum(1 for p in board.piece_map().values() if p.color == chess.WHITE)
        black_count = sum(1 for p in board.piece_map().values() if p.color == chess.BLACK)
        
        # If more white pieces are missing, it's likely black's turn
        board.turn = chess.WHITE if white_count >= black_count else chess.BLACK
        
        return board

if __name__ == "__main__":
    print("Starting Camera Chess Analyzer")
    print("Position your chess board in front of the camera")
    print("Press ESC to quit")
    
    try:
        # Use the basic analyzer
        analyzer = CameraChessAnalyzer()
        
        # For better piece recognition, use the advanced analyzer (requires ML model)
        # analyzer = AdvancedChessAnalyzer()
        
        analyzer.run()
    except Exception as e:
        print(f"Error: {e}")