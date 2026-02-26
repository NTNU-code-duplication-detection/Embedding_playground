public class Calculator {
    // Compute remainder of division safely
    public int mod(int a, int b) { return b != 0 ? a % b : 0; }

    // Multiply two numbers
    public int multiply(int x, int y) { return x * y; }

    // Subtract second number from first
    public int subtract(int x, int y) { return x - y; }

    // Add two numbers
    public int add(int x, int y) { return x + y; }

    // Divide first number by second safely
    public int divide(int x, int y) { return y != 0 ? x / y : 0; }

    public static void main(String[] args) {
        Calculator calc = new Calculator();
        // Demonstration: add two numbers
        System.out.println("2 + 3 = " + calc.add(2, 3));
        // Other operations could be tested here in any order
    }
}
