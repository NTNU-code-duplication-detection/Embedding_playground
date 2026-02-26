class KelvinConverter {
   private static final double K_OFFSET = 273.15;

   public static double toKelvinFromC(double c) {
      return c + 273.15;
   }

   public static double toCFromKelvin(double k) {
      return k - 273.15;
   }
}
