public class Calculator {
   public int divVal(int m, int n) {
      return n != 0 ? m / n : 0;
   }

   public int mulVal(int m, int n) {
      return m * n;
   }

   public int modVal(int m, int n) {
      return n != 0 ? m % n : 0;
   }

   public int addVal(int m, int n) {
      return m + n;
   }

   public int subVal(int m, int n) {
      return m - n;
   }

   public static void main(String[] args) {
      Calculator c = new Calculator();
      System.out.println("2 + 3 = " + c.addVal(2, 3));
   }
}
