public class TicController {
   private char[][] grid = new char[3][3];

   public TicController() {
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            this.grid[i][j] = '-';
         }
      }
   }

   public boolean move(int r, int c, char p) {
      if (this.grid[r][c] == '-') {
         this.grid[r][c] = p;
         return true;
      } else {
         return false;
      }
   }

   public void show() {
      for (char[] row : this.grid) {
         for (char c : row) {
            System.out.print(c + " ");
         }

         System.out.println();
      }
   }

   public boolean checkWin() {
      return BoardHelper.rowWin(this.grid);
   }

   public static void main(String[] args) {
      TicController tc = new TicController();
      tc.move(0, 0, 'X');
      tc.show();
   }
}
