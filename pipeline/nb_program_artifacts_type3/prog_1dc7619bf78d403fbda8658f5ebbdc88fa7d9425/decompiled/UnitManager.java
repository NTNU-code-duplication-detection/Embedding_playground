public class UnitManager {
   public double mToKm(double m) {
      return ScaleHelper.down(m);
   }

   public double kmToM(double km) {
      return ScaleHelper.up(km);
   }

   public double gToKg(double g) {
      return ScaleHelper.down(g);
   }

   public double kgToG(double kg) {
      return ScaleHelper.up(kg);
   }

   public double lToMl(double l) {
      return ScaleHelper.up(l);
   }

   public static void main(String[] args) {
      UnitManager um = new UnitManager();
      System.out.println("5000m = " + um.mToKm(5000.0) + " km");
   }
}
