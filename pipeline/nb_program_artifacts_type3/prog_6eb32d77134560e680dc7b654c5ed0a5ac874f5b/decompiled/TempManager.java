public class TempManager {
   public double cToF(double c) {
      return c * 9.0 / 5.0 + 32.0;
   }

   public double fToC(double f) {
      return (f - 32.0) * 5.0 / 9.0;
   }

   public double fToK(double f) {
      return KelvinConverter.toKelvinFromC(this.fToC(f));
   }

   public static void main(String[] args) {
      TempManager tm = new TempManager();
      System.out.println("0°C = " + tm.cToF(0.0) + "°F");
   }
}
