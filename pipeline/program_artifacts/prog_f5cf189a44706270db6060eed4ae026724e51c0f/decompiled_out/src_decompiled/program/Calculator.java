public class Calculator {
   public int difference(int var1, int var2) {
      return var1 - var2;
   }

   public int safeDivide(int var1, int var2) {
      return var2 != 0 ? var1 / var2 : 0;
   }

   public int product(int var1, int var2) {
      return var1 * var2;
   }

   public int sum(int var1, int var2) {
      return var1 + var2;
   }

   public int remainder(int var1, int var2) {
      return var2 != 0 ? var1 % var2 : 0;
   }

   public static void main(String[] var0) {
      Calculator var1 = new Calculator();
      System.out.println("2 + 3 = " + var1.sum(2, 3));
   }
}
