public class Calculator {
    public int add(int a, int b) { return a + b; }
    public int subtract(int a, int b) { return a - b; }
    public int multiply(int a, int b) { return a * b; }
    public int divide(int a, int b) { return b != 0 ? a / b : 0; }
    public int modulo(int a, int b) { return b != 0 ? a % b : 0; }

    public static void main(String[] args) {
        Calculator calc = new Calculator();
        System.out.println("2 + 3 = " + calc.add(2, 3));
    }
}
