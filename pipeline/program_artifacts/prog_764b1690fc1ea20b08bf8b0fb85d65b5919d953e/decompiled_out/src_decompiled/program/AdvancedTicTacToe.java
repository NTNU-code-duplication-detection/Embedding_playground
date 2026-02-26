public class AdvancedTicTacToe {
   private char[][] board = this.createBoard();

   private char[][] createBoard() {
      char[][] var1 = new char[3][3];

      for (int var2 = 0; var2 < 3; var2++) {
         for (int var3 = 0; var3 < 3; var3++) {
            var1[var2][var3] = '-';
         }
      }

      return var1;
   }

   public boolean makeMove(int var1, int var2, char var3) {
      if (this.board[var1][var2] != '-') {
         return false;
      } else {
         this.board[var1][var2] = var3;
         return true;
      }
   }

   public boolean checkHorizontalWin() {
      for (int var1 = 0; var1 < 3; var1++) {
         if (this.board[var1][0] == this.board[var1][1] && this.board[var1][1] == this.board[var1][2] && this.board[var1][0] != '-') {
            return true;
         }
      }

      return false;
   }

   public void display() {
      for (char[] var4 : this.board) {
         for (char var8 : var4) {
            System.out.print(var8 + " ");
         }

         System.out.println();
      }
   }

   public static void main(String[] var0) {
      AdvancedTicTacToe var1 = new AdvancedTicTacToe();
      var1.makeMove(0, 0, 'X');
      var1.display();
   }
}
