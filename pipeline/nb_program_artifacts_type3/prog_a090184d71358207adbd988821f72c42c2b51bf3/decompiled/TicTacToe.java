public class TicTacToe {
   private char[][] board = new char[3][3];

   public TicTacToe() {
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            this.board[i][j] = '-';
         }
      }
   }

   public boolean makeMove(int row, int col, char player) {
      if (this.board[row][col] == '-') {
         this.board[row][col] = player;
         return true;
      } else {
         return false;
      }
   }

   public void printBoard() {
      for (char[] row : this.board) {
         for (char c : row) {
            System.out.print(c + " ");
         }

         System.out.println();
      }
   }

   public boolean checkWin() {
      for (int i = 0; i < 3; i++) {
         if (this.board[i][0] == this.board[i][1] && this.board[i][1] == this.board[i][2] && this.board[i][0] != '-') {
            return true;
         }
      }

      return false;
   }

   public void reset() {
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            this.board[i][j] = '-';
         }
      }
   }

   public static void main(String[] args) {
      TicTacToe game = new TicTacToe();
      game.makeMove(0, 0, 'X');
      game.printBoard();
   }
}
