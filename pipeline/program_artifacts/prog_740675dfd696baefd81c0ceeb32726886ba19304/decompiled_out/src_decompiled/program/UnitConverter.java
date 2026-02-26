public class UnitConverter {
   public double metersToKilometers(double var1) {
      return var1 / 1000.0;
   }

   public double kilometersToMeters(double var1) {
      return var1 * 1000.0;
   }

   public double gramsToKilograms(double var1) {
      return var1 / 1000.0;
   }

   public double kilogramsToGrams(double var1) {
      return var1 * 1000.0;
   }

   public double litersToMilliliters(double var1) {
      return var1 * 1000.0;
   }

   public static void main(String[] var0) {
      UnitConverter var1 = new UnitConverter();
      System.out.println("5000m = " + var1.metersToKilometers(5000.0) + " km");
   }
}
