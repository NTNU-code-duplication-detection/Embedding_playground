public class NumberPalindromeManager {

    // Core comparison logic
    private boolean compare(String original) {
        String reversed = new StringBuilder(original).reverse().toString();
        return original.equals(reversed);
    }

    public boolean isPalindromeString(String text) {
        return compare(text);
    }

    public boolean isPalindromeNumber(int num) {
        return compare(Integer.toString(num));
    }

    public static void main(String[] args) {
        NumberPalindromeManager npm = new NumberPalindromeManager();
        System.out.println("121 palindrome? " + npm.isPalindromeNumber(121));
    }
}
