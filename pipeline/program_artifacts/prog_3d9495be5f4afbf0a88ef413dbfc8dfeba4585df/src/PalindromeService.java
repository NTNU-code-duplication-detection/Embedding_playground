class StringHelper {
    public static boolean isPalindrome(String value) {
        String reversed = new StringBuilder(value).reverse().toString();
        return value.equals(reversed);
    }
}

public class PalindromeService {

    public boolean checkString(String s) {
        return StringHelper.isPalindrome(s);
    }

    public boolean checkNumber(int n) {
        return checkString(String.valueOf(n));
    }

    public static void main(String[] args) {
        PalindromeService service = new PalindromeService();
        System.out.println("121 palindrome? " + service.checkNumber(121));
    }
}
