public class UnitManager {
   public double mToKm(double var1) {
      return ScaleHelper.down(var1);
   }

   public double kmToM(double var1) {
      return ScaleHelper.up(var1);
   }

   public double gToKg(double var1) {
      return ScaleHelper.down(var1);
   }

   public double kgToG(double var1) {
      return ScaleHelper.up(var1);
   }

   public double lToMl(double var1) {
      return ScaleHelper.up(var1);
   }

   public static void main(String[] var0) {
      UnitManager var1 = new UnitManager();
      System.out.println("5000m = " + var1.mToKm(5000.0) + " km");
   }
}
