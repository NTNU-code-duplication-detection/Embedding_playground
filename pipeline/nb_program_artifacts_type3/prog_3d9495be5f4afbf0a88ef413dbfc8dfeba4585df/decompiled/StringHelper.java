class StringHelper {
   public static boolean isPalindrome(String value) {
      String reversed = new StringBuilder(value).reverse().toString();
      return value.equals(reversed);
   }
}
