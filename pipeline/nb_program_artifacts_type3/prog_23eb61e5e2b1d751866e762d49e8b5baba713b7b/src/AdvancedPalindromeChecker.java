public class AdvancedPalindromeChecker {

    // Helper to reverse strings
    private String reverse(String input) {
        return new StringBuilder(input).reverse().toString();
    }

    public boolean isTextPalindrome(String text) {
        return text.equals(reverse(text));
    }

    public boolean isNumberPalindrome(int number) {
        return isTextPalindrome(String.valueOf(number));
    }

    public static void main(String[] args) {
        AdvancedPalindromeChecker apc = new AdvancedPalindromeChecker();
        System.out.println("121 palindrome? " + apc.isNumberPalindrome(121));
    }
}
