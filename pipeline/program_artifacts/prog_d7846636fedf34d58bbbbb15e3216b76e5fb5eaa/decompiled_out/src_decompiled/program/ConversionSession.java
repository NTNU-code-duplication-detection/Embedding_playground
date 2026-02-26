public class ConversionSession {
   private static final double MULTIPLIER = 1000.0;

   public double convertDown(double var1) {
      return var1 / 1000.0;
   }

   public double convertUp(double var1) {
      return var1 * 1000.0;
   }

   public static void main(String[] var0) {
      ConversionSession var1 = new ConversionSession();
      System.out.println("5000m = " + var1.convertDown(5000.0) + " km");
   }
}
