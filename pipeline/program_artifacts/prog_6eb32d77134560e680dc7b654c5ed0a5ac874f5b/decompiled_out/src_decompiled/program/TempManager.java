public class TempManager {
   public double cToF(double var1) {
      return var1 * 9.0 / 5.0 + 32.0;
   }

   public double fToC(double var1) {
      return (var1 - 32.0) * 5.0 / 9.0;
   }

   public double fToK(double var1) {
      return KelvinConverter.toKelvinFromC(this.fToC(var1));
   }

   public static void main(String[] var0) {
      TempManager var1 = new TempManager();
      System.out.println("0°C = " + var1.cToF(0.0) + "°F");
   }
}
