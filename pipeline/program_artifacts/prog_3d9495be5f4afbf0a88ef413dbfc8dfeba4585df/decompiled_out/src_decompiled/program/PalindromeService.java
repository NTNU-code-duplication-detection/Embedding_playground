public class PalindromeService {
   public boolean checkString(String var1) {
      return StringHelper.isPalindrome(var1);
   }

   public boolean checkNumber(int var1) {
      return this.checkString(String.valueOf(var1));
   }

   public static void main(String[] var0) {
      PalindromeService var1 = new PalindromeService();
      System.out.println("121 palindrome? " + var1.checkNumber(121));
   }
}
