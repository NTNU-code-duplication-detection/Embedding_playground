public class AdvancedTicTacToe {
   private char[][] board = this.createBoard();

   private char[][] createBoard() {
      char[][] b = new char[3][3];

      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            b[i][j] = '-';
         }
      }

      return b;
   }

   public boolean makeMove(int row, int col, char player) {
      if (this.board[row][col] != '-') {
         return false;
      } else {
         this.board[row][col] = player;
         return true;
      }
   }

   public boolean checkHorizontalWin() {
      for (int i = 0; i < 3; i++) {
         if (this.board[i][0] == this.board[i][1] && this.board[i][1] == this.board[i][2] && this.board[i][0] != '-') {
            return true;
         }
      }

      return false;
   }

   public void display() {
      for (char[] r : this.board) {
         for (char c : r) {
            System.out.print(c + " ");
         }

         System.out.println();
      }
   }

   public static void main(String[] args) {
      AdvancedTicTacToe game = new AdvancedTicTacToe();
      game.makeMove(0, 0, 'X');
      game.display();
   }
}
