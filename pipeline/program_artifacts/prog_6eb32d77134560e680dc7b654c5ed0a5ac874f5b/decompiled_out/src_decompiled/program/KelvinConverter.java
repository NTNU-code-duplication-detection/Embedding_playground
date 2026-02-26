class KelvinConverter {
   private static final double K_OFFSET = 273.15;

   public static double toKelvinFromC(double var0) {
      return var0 + 273.15;
   }

   public static double toCFromKelvin(double var0) {
      return var0 - 273.15;
   }
}
