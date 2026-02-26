public class TemperatureConverter {
   public double celsiusToFahrenheit(double c) {
      return c * 9.0 / 5.0 + 32.0;
   }

   public double fahrenheitToCelsius(double f) {
      return (f - 32.0) * 5.0 / 9.0;
   }

   public double celsiusToKelvin(double c) {
      return c + 273.15;
   }

   public double kelvinToCelsius(double k) {
      return k - 273.15;
   }

   public double fahrenheitToKelvin(double f) {
      return (f - 32.0) * 5.0 / 9.0 + 273.15;
   }

   public static void main(String[] args) {
      TemperatureConverter t = new TemperatureConverter();
      System.out.println("0°C = " + t.celsiusToFahrenheit(0.0) + "°F");
   }
}
