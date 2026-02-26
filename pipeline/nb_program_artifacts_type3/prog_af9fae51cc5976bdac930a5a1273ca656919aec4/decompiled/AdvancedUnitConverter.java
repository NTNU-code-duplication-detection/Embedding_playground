public class AdvancedUnitConverter {
   private static final double SCALE = 1000.0;

   private double scale(double value, boolean up) {
      return up ? value * 1000.0 : value / 1000.0;
   }

   public double metersToKilometers(double m) {
      return this.scale(m, false);
   }

   public double kilometersToMeters(double km) {
      return this.scale(km, true);
   }

   public double gramsToKilograms(double g) {
      return this.scale(g, false);
   }

   public double kilogramsToGrams(double kg) {
      return this.scale(kg, true);
   }

   public double litersToMilliliters(double l) {
      return this.scale(l, true);
   }

   public static void main(String[] args) {
      AdvancedUnitConverter auc = new AdvancedUnitConverter();
      System.out.println("5000m = " + auc.metersToKilometers(5000.0) + " km");
   }
}
