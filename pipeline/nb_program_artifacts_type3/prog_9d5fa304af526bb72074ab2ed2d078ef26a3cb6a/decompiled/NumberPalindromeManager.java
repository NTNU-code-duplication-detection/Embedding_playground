public class NumberPalindromeManager {
   private boolean compare(String original) {
      String reversed = new StringBuilder(original).reverse().toString();
      return original.equals(reversed);
   }

   public boolean isPalindromeString(String text) {
      return this.compare(text);
   }

   public boolean isPalindromeNumber(int num) {
      return this.compare(Integer.toString(num));
   }

   public static void main(String[] args) {
      NumberPalindromeManager npm = new NumberPalindromeManager();
      System.out.println("121 palindrome? " + npm.isPalindromeNumber(121));
   }
}
