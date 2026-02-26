public class TemperatureConverter {
   public double celsiusToFahrenheit(double var1) {
      return var1 * 9.0 / 5.0 + 32.0;
   }

   public double fahrenheitToCelsius(double var1) {
      return (var1 - 32.0) * 5.0 / 9.0;
   }

   public double celsiusToKelvin(double var1) {
      return var1 + 273.15;
   }

   public double kelvinToCelsius(double var1) {
      return var1 - 273.15;
   }

   public double fahrenheitToKelvin(double var1) {
      return (var1 - 32.0) * 5.0 / 9.0 + 273.15;
   }

   public static void main(String[] var0) {
      TemperatureConverter var1 = new TemperatureConverter();
      System.out.println("0°C = " + var1.celsiusToFahrenheit(0.0) + "°F");
   }
}
