public class GameSession {
   private char[][] board = new char[3][3];

   public GameSession() {
      this.reset();
   }

   public void reset() {
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            this.board[i][j] = '-';
         }
      }
   }

   public boolean placeMark(int row, int col, char mark) {
      if (this.board[row][col] != '-') {
         return false;
      } else {
         this.board[row][col] = mark;
         return true;
      }
   }

   private boolean rowWin() {
      for (int i = 0; i < 3; i++) {
         if (this.board[i][0] == this.board[i][1] && this.board[i][1] == this.board[i][2] && this.board[i][0] != '-') {
            return true;
         }
      }

      return false;
   }

   public void printBoard() {
      for (char[] row : this.board) {
         for (char c : row) {
            System.out.print(c + " ");
         }

         System.out.println();
      }
   }

   public static void main(String[] args) {
      GameSession gs = new GameSession();
      gs.placeMark(0, 0, 'X');
      gs.printBoard();
   }
}
