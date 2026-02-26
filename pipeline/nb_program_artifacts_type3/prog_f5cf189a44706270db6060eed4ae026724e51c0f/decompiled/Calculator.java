public class Calculator {
   public int difference(int a, int b) {
      return a - b;
   }

   public int safeDivide(int a, int b) {
      return b != 0 ? a / b : 0;
   }

   public int product(int x, int y) {
      return x * y;
   }

   public int sum(int x, int y) {
      return x + y;
   }

   public int remainder(int x, int y) {
      return y != 0 ? x % y : 0;
   }

   public static void main(String[] args) {
      Calculator calc = new Calculator();
      System.out.println("2 + 3 = " + calc.sum(2, 3));
   }
}
