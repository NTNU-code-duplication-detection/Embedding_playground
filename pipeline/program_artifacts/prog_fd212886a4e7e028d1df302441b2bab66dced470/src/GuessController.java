import java.util.Random;
import java.util.Scanner;

class GuessEngine {
    private int secret = new Random().nextInt(100) + 1;

    public String compare(int input) {
        if (input == secret) return "Correct!";
        return input < secret ? "Too low" : "Too high";
    }
}

public class GuessController {
    private GuessEngine engine = new GuessEngine();
    private int attempts = 0;

    public String handleGuess(int value) {
        attempts++;
        return engine.compare(value);
    }

    public static void main(String[] args) {
        GuessController controller = new GuessController();
        Scanner sc = new Scanner(System.in);
        System.out.println("Guess a number between 1-100:");
        System.out.println(controller.handleGuess(sc.nextInt()));
    }
}
