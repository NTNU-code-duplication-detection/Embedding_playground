public class Calculator {
   public int divVal(int var1, int var2) {
      return var2 != 0 ? var1 / var2 : 0;
   }

   public int mulVal(int var1, int var2) {
      return var1 * var2;
   }

   public int modVal(int var1, int var2) {
      return var2 != 0 ? var1 % var2 : 0;
   }

   public int addVal(int var1, int var2) {
      return var1 + var2;
   }

   public int subVal(int var1, int var2) {
      return var1 - var2;
   }

   public static void main(String[] var0) {
      Calculator var1 = new Calculator();
      System.out.println("2 + 3 = " + var1.addVal(2, 3));
   }
}
