public class MultiTempConverter {
    private static final double KELVIN = 273.15;

    public double celsiusToF(double tempC) { return tempC * 9/5 + 32; }
    public double fahrenheitToC(double tempF) { return (tempF - 32) * 5/9; }

    public double celsiusToK(double tempC) { return tempC + KELVIN; }
    public double kelvinToC(double tempK) { return tempK - KELVIN; }
    public double fahrenheitToK(double tempF) { return celsiusToK(fahrenheitToC(tempF)); }

    public static void main(String[] args) {
        MultiTempConverter converter = new MultiTempConverter();
        System.out.println("0°C = " + converter.celsiusToF(0) + "°F");
    }
}
