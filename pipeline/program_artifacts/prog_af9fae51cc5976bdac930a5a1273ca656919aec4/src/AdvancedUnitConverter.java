public class AdvancedUnitConverter {
    private static final double SCALE = 1000;

    // Generic scaling helper
    private double scale(double value, boolean up) {
        return up ? value * SCALE : value / SCALE;
    }

    public double metersToKilometers(double m) { return scale(m, false); }
    public double kilometersToMeters(double km) { return scale(km, true); }
    public double gramsToKilograms(double g) { return scale(g, false); }
    public double kilogramsToGrams(double kg) { return scale(kg, true); }
    public double litersToMilliliters(double l) { return scale(l, true); }

    public static void main(String[] args) {
        AdvancedUnitConverter auc = new AdvancedUnitConverter();
        System.out.println("5000m = " + auc.metersToKilometers(5000) + " km");
    }
}
