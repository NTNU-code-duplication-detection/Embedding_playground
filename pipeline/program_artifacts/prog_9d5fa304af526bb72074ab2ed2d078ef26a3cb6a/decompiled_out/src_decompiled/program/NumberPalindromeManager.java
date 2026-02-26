public class NumberPalindromeManager {
   private boolean compare(String var1) {
      String var2 = new StringBuilder(var1).reverse().toString();
      return var1.equals(var2);
   }

   public boolean isPalindromeString(String var1) {
      return this.compare(var1);
   }

   public boolean isPalindromeNumber(int var1) {
      return this.compare(Integer.toString(var1));
   }

   public static void main(String[] var0) {
      NumberPalindromeManager var1 = new NumberPalindromeManager();
      System.out.println("121 palindrome? " + var1.isPalindromeNumber(121));
   }
}
