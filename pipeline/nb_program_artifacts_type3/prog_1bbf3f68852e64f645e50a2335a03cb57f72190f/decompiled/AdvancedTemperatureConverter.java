public class AdvancedTemperatureConverter {
   private static final double KELVIN_BASE = 273.15;

   public double celsiusToFahrenheit(double c) {
      return c * 9.0 / 5.0 + 32.0;
   }

   public double fahrenheitToCelsius(double f) {
      return (f - 32.0) * 5.0 / 9.0;
   }

   private double offsetKelvin(double temp, boolean add) {
      return add ? temp + 273.15 : temp - 273.15;
   }

   public double celsiusToKelvin(double c) {
      return this.offsetKelvin(c, true);
   }

   public double kelvinToCelsius(double k) {
      return this.offsetKelvin(k, false);
   }

   public double fahrenheitToKelvin(double f) {
      return this.offsetKelvin(this.fahrenheitToCelsius(f), true);
   }

   public static void main(String[] args) {
      AdvancedTemperatureConverter atc = new AdvancedTemperatureConverter();
      System.out.println("0°C = " + atc.celsiusToFahrenheit(0.0) + "°F");
   }
}
