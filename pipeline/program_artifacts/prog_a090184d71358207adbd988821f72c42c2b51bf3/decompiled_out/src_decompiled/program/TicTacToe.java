public class TicTacToe {
   private char[][] board = new char[3][3];

   public TicTacToe() {
      for (int var1 = 0; var1 < 3; var1++) {
         for (int var2 = 0; var2 < 3; var2++) {
            this.board[var1][var2] = '-';
         }
      }
   }

   public boolean makeMove(int var1, int var2, char var3) {
      if (this.board[var1][var2] == '-') {
         this.board[var1][var2] = var3;
         return true;
      } else {
         return false;
      }
   }

   public void printBoard() {
      for (char[] var4 : this.board) {
         for (char var8 : var4) {
            System.out.print(var8 + " ");
         }

         System.out.println();
      }
   }

   public boolean checkWin() {
      for (int var1 = 0; var1 < 3; var1++) {
         if (this.board[var1][0] == this.board[var1][1] && this.board[var1][1] == this.board[var1][2] && this.board[var1][0] != '-') {
            return true;
         }
      }

      return false;
   }

   public void reset() {
      for (int var1 = 0; var1 < 3; var1++) {
         for (int var2 = 0; var2 < 3; var2++) {
            this.board[var1][var2] = '-';
         }
      }
   }

   public static void main(String[] var0) {
      TicTacToe var1 = new TicTacToe();
      var1.makeMove(0, 0, 'X');
      var1.printBoard();
   }
}
