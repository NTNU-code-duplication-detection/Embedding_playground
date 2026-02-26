public class Calculator {
   public int add(int var1, int var2) {
      return var1 + var2;
   }

   public int subtract(int var1, int var2) {
      return var1 - var2;
   }

   public int multiply(int var1, int var2) {
      return var1 * var2;
   }

   public int divide(int var1, int var2) {
      return var2 != 0 ? var1 / var2 : 0;
   }

   public int modulo(int var1, int var2) {
      return var2 != 0 ? var1 % var2 : 0;
   }

   public static void main(String[] var0) {
      Calculator var1 = new Calculator();
      System.out.println("2 + 3 = " + var1.add(2, 3));
   }
}
