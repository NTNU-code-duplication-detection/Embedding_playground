public class AdvancedTemperatureConverter {
    private static final double KELVIN_BASE = 273.15;

    public double celsiusToFahrenheit(double c) { return c * 9/5 + 32; }
    public double fahrenheitToCelsius(double f) { return (f - 32) * 5/9; }

    // Helper for Kelvin conversions
    private double offsetKelvin(double temp, boolean add) { return add ? temp + KELVIN_BASE : temp - KELVIN_BASE; }

    public double celsiusToKelvin(double c) { return offsetKelvin(c, true); }
    public double kelvinToCelsius(double k) { return offsetKelvin(k, false); }
    public double fahrenheitToKelvin(double f) { return offsetKelvin(fahrenheitToCelsius(f), true); }

    public static void main(String[] args) {
        AdvancedTemperatureConverter atc = new AdvancedTemperatureConverter();
        System.out.println("0°C = " + atc.celsiusToFahrenheit(0) + "°F");
    }
}
