class KelvinConverter {
    private static final double K_OFFSET = 273.15;

    public static double toKelvinFromC(double c) { return c + K_OFFSET; }
    public static double toCFromKelvin(double k) { return k - K_OFFSET; }
}

public class TempManager {
    public double cToF(double c) { return c * 9/5 + 32; }
    public double fToC(double f) { return (f - 32) * 5/9; }
    public double fToK(double f) { return KelvinConverter.toKelvinFromC(fToC(f)); }

    public static void main(String[] args) {
        TempManager tm = new TempManager();
        System.out.println("0°C = " + tm.cToF(0) + "°F");
    }
}
