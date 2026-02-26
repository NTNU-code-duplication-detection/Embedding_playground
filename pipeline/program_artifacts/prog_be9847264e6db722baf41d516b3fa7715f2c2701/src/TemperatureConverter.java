public class TemperatureConverter {
    public double celsiusToFahrenheit(double c) { return c * 9/5 + 32; }
    public double fahrenheitToCelsius(double f) { return (f - 32) * 5/9; }
    public double celsiusToKelvin(double c) { return c + 273.15; }
    public double kelvinToCelsius(double k) { return k - 273.15; }
    public double fahrenheitToKelvin(double f) { return (f - 32) * 5/9 + 273.15; }

    public static void main(String[] args) {
        TemperatureConverter t = new TemperatureConverter();
        System.out.println("0°C = " + t.celsiusToFahrenheit(0) + "°F");
    }
}
