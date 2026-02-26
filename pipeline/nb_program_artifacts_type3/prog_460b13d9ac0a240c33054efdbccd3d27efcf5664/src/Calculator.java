public class Calculator {
    public int divVal(int m, int n) { return n != 0 ? m / n : 0; }   // divide safely
    public int mulVal(int m, int n) { return m * n; }                 // multiply
    public int modVal(int m, int n) { return n != 0 ? m % n : 0; }   // remainder
    public int addVal(int m, int n) { return m + n; }                 // sum
    public int subVal(int m, int n) { return m - n; }                 // subtract

    public static void main(String[] args) {
        Calculator c = new Calculator();
        System.out.println("2 + 3 = " + c.addVal(2, 3)); // quick test
    }
}
