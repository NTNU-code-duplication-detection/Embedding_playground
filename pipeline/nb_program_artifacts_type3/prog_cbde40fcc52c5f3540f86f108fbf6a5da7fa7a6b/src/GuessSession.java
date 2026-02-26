import java.util.Random;
import java.util.Scanner;

public class GuessSession {
    private int attempts = 0;
    private int number;

    public GuessSession() {
        number = new Random().nextInt(100) + 1;
    }

    private String result(int guess) {
        if (guess == number) return "Correct!";
        return guess < number ? "Too low" : "Too high";
    }

    public String submit(int guess) {
        attempts++;
        return result(guess);
    }

    public static void main(String[] args) {
        GuessSession session = new GuessSession();
        Scanner sc = new Scanner(System.in);
        System.out.println("Guess a number between 1-100:");
        System.out.println(session.submit(sc.nextInt()));
    }
}
