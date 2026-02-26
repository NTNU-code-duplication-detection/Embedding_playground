public class MultiTempConverter {
   private static final double KELVIN = 273.15;

   public double celsiusToF(double tempC) {
      return tempC * 9.0 / 5.0 + 32.0;
   }

   public double fahrenheitToC(double tempF) {
      return (tempF - 32.0) * 5.0 / 9.0;
   }

   public double celsiusToK(double tempC) {
      return tempC + 273.15;
   }

   public double kelvinToC(double tempK) {
      return tempK - 273.15;
   }

   public double fahrenheitToK(double tempF) {
      return this.celsiusToK(this.fahrenheitToC(tempF));
   }

   public static void main(String[] args) {
      MultiTempConverter converter = new MultiTempConverter();
      System.out.println("0°C = " + converter.celsiusToF(0.0) + "°F");
   }
}
