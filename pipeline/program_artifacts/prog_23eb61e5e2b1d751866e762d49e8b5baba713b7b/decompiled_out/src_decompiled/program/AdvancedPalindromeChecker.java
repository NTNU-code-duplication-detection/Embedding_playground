public class AdvancedPalindromeChecker {
   private String reverse(String var1) {
      return new StringBuilder(var1).reverse().toString();
   }

   public boolean isTextPalindrome(String var1) {
      return var1.equals(this.reverse(var1));
   }

   public boolean isNumberPalindrome(int var1) {
      return this.isTextPalindrome(String.valueOf(var1));
   }

   public static void main(String[] var0) {
      AdvancedPalindromeChecker var1 = new AdvancedPalindromeChecker();
      System.out.println("121 palindrome? " + var1.isNumberPalindrome(121));
   }
}
