public class UnitConverter {
   public double metersToKilometers(double m) {
      return m / 1000.0;
   }

   public double kilometersToMeters(double km) {
      return km * 1000.0;
   }

   public double gramsToKilograms(double g) {
      return g / 1000.0;
   }

   public double kilogramsToGrams(double kg) {
      return kg * 1000.0;
   }

   public double litersToMilliliters(double l) {
      return l * 1000.0;
   }

   public static void main(String[] args) {
      UnitConverter uc = new UnitConverter();
      System.out.println("5000m = " + uc.metersToKilometers(5000.0) + " km");
   }
}
