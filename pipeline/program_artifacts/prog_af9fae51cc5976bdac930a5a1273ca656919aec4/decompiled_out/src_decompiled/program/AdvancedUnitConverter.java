public class AdvancedUnitConverter {
   private static final double SCALE = 1000.0;

   private double scale(double var1, boolean var3) {
      return var3 ? var1 * 1000.0 : var1 / 1000.0;
   }

   public double metersToKilometers(double var1) {
      return this.scale(var1, false);
   }

   public double kilometersToMeters(double var1) {
      return this.scale(var1, true);
   }

   public double gramsToKilograms(double var1) {
      return this.scale(var1, false);
   }

   public double kilogramsToGrams(double var1) {
      return this.scale(var1, true);
   }

   public double litersToMilliliters(double var1) {
      return this.scale(var1, true);
   }

   public static void main(String[] var0) {
      AdvancedUnitConverter var1 = new AdvancedUnitConverter();
      System.out.println("5000m = " + var1.metersToKilometers(5000.0) + " km");
   }
}
