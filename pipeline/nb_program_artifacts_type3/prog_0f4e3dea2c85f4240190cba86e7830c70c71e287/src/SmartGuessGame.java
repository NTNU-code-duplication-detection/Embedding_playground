import java.util.Random;
import java.util.Scanner;

public class SmartGuessGame {
    private int target;
    private int attempts;

    public SmartGuessGame() {
        target = generateNumber();
    }

    // Helper method for random number generation
    private int generateNumber() {
        return new Random().nextInt(100) + 1;
    }

    public String guess(int value) {
        attempts++;
        if (value == target) return "Correct!";
        return value < target ? "Too low" : "Too high";
    }

    public static void main(String[] args) {
        SmartGuessGame game = new SmartGuessGame();
        Scanner sc = new Scanner(System.in);
        System.out.println("Guess a number between 1-100:");
        System.out.println(game.guess(sc.nextInt()));
    }
}
