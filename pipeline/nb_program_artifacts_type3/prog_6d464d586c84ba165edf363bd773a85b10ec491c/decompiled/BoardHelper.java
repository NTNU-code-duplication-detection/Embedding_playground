class BoardHelper {
   public static boolean rowWin(char[][] board) {
      for (int i = 0; i < 3; i++) {
         if (board[i][0] == board[i][1] && board[i][1] == board[i][2] && board[i][0] != '-') {
            return true;
         }
      }

      return false;
   }
}
