public class AdvancedTemperatureConverter {
   private static final double KELVIN_BASE = 273.15;

   public double celsiusToFahrenheit(double var1) {
      return var1 * 9.0 / 5.0 + 32.0;
   }

   public double fahrenheitToCelsius(double var1) {
      return (var1 - 32.0) * 5.0 / 9.0;
   }

   private double offsetKelvin(double var1, boolean var3) {
      return var3 ? var1 + 273.15 : var1 - 273.15;
   }

   public double celsiusToKelvin(double var1) {
      return this.offsetKelvin(var1, true);
   }

   public double kelvinToCelsius(double var1) {
      return this.offsetKelvin(var1, false);
   }

   public double fahrenheitToKelvin(double var1) {
      return this.offsetKelvin(this.fahrenheitToCelsius(var1), true);
   }

   public static void main(String[] var0) {
      AdvancedTemperatureConverter var1 = new AdvancedTemperatureConverter();
      System.out.println("0°C = " + var1.celsiusToFahrenheit(0.0) + "°F");
   }
}
