public class GameSession {
   private char[][] board = new char[3][3];

   public GameSession() {
      this.reset();
   }

   public void reset() {
      for (int var1 = 0; var1 < 3; var1++) {
         for (int var2 = 0; var2 < 3; var2++) {
            this.board[var1][var2] = '-';
         }
      }
   }

   public boolean placeMark(int var1, int var2, char var3) {
      if (this.board[var1][var2] != '-') {
         return false;
      } else {
         this.board[var1][var2] = var3;
         return true;
      }
   }

   private boolean rowWin() {
      for (int var1 = 0; var1 < 3; var1++) {
         if (this.board[var1][0] == this.board[var1][1] && this.board[var1][1] == this.board[var1][2] && this.board[var1][0] != '-') {
            return true;
         }
      }

      return false;
   }

   public void printBoard() {
      for (char[] var4 : this.board) {
         for (char var8 : var4) {
            System.out.print(var8 + " ");
         }

         System.out.println();
      }
   }

   public static void main(String[] var0) {
      GameSession var1 = new GameSession();
      var1.placeMark(0, 0, 'X');
      var1.printBoard();
   }
}
