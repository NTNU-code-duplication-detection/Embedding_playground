class StringHelper {
   public static boolean isPalindrome(String var0) {
      String var1 = new StringBuilder(var0).reverse().toString();
      return var0.equals(var1);
   }
}
