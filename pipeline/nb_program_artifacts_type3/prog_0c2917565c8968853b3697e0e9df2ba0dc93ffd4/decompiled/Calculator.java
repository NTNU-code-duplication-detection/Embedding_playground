public class Calculator {
   public int mod(int a, int b) {
      return b != 0 ? a % b : 0;
   }

   public int multiply(int x, int y) {
      return x * y;
   }

   public int subtract(int x, int y) {
      return x - y;
   }

   public int add(int x, int y) {
      return x + y;
   }

   public int divide(int x, int y) {
      return y != 0 ? x / y : 0;
   }

   public static void main(String[] args) {
      Calculator calc = new Calculator();
      System.out.println("2 + 3 = " + calc.add(2, 3));
   }
}
