public class PalindromeChecker {
   public boolean isPalindrome(String str) {
      String reversed = new StringBuilder(str).reverse().toString();
      return str.equals(reversed);
   }

   public boolean isPalindrome(int num) {
      return this.isPalindrome(String.valueOf(num));
   }

   public static void main(String[] args) {
      PalindromeChecker pc = new PalindromeChecker();
      System.out.println("121 palindrome? " + pc.isPalindrome(121));
   }
}
