public class Calculator {
    // Subtract b from a
    public int difference(int a, int b) { return a - b; }

    // Safe division
    public int safeDivide(int a, int b) { return b != 0 ? a / b : 0; }

    // Compute product
    public int product(int x, int y) { return x * y; }

    // Add two numbers together
    public int sum(int x, int y) { return x + y; }

    // Find remainder
    public int remainder(int x, int y) { return y != 0 ? x % y : 0; }

    public static void main(String[] args) {
        Calculator calc = new Calculator();
        // Adding numbers example
        System.out.println("2 + 3 = " + calc.sum(2, 3));
    }
}
