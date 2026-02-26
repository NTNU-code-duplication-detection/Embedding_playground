public class MultiTempConverter {
   private static final double KELVIN = 273.15;

   public double celsiusToF(double var1) {
      return var1 * 9.0 / 5.0 + 32.0;
   }

   public double fahrenheitToC(double var1) {
      return (var1 - 32.0) * 5.0 / 9.0;
   }

   public double celsiusToK(double var1) {
      return var1 + 273.15;
   }

   public double kelvinToC(double var1) {
      return var1 - 273.15;
   }

   public double fahrenheitToK(double var1) {
      return this.celsiusToK(this.fahrenheitToC(var1));
   }

   public static void main(String[] var0) {
      MultiTempConverter var1 = new MultiTempConverter();
      System.out.println("0°C = " + var1.celsiusToF(0.0) + "°F");
   }
}
