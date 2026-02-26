public class TicController {
   private char[][] grid = new char[3][3];

   public TicController() {
      for (int var1 = 0; var1 < 3; var1++) {
         for (int var2 = 0; var2 < 3; var2++) {
            this.grid[var1][var2] = '-';
         }
      }
   }

   public boolean move(int var1, int var2, char var3) {
      if (this.grid[var1][var2] == '-') {
         this.grid[var1][var2] = var3;
         return true;
      } else {
         return false;
      }
   }

   public void show() {
      for (char[] var4 : this.grid) {
         for (char var8 : var4) {
            System.out.print(var8 + " ");
         }

         System.out.println();
      }
   }

   public boolean checkWin() {
      return BoardHelper.rowWin(this.grid);
   }

   public static void main(String[] var0) {
      TicController var1 = new TicController();
      var1.move(0, 0, 'X');
      var1.show();
   }
}
