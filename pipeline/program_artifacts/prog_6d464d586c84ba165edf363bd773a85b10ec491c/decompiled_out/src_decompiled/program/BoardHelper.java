class BoardHelper {
   public static boolean rowWin(char[][] var0) {
      for (int var1 = 0; var1 < 3; var1++) {
         if (var0[var1][0] == var0[var1][1] && var0[var1][1] == var0[var1][2] && var0[var1][0] != '-') {
            return true;
         }
      }

      return false;
   }
}
