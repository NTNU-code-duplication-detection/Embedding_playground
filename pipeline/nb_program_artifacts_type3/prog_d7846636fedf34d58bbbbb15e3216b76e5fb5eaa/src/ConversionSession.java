public class ConversionSession {

    private static final double MULTIPLIER = 1000;

    public double convertDown(double value) {
        return value / MULTIPLIER;
    }

    public double convertUp(double value) {
        return value * MULTIPLIER;
    }

    public static void main(String[] args) {
        ConversionSession cs = new ConversionSession();
        System.out.println("5000m = " + cs.convertDown(5000) + " km");
    }
}
