public class PalindromeChecker {
   public boolean isPalindrome(String var1) {
      String var2 = new StringBuilder(var1).reverse().toString();
      return var1.equals(var2);
   }

   public boolean isPalindrome(int var1) {
      return this.isPalindrome(String.valueOf(var1));
   }

   public static void main(String[] var0) {
      PalindromeChecker var1 = new PalindromeChecker();
      System.out.println("121 palindrome? " + var1.isPalindrome(121));
   }
}
