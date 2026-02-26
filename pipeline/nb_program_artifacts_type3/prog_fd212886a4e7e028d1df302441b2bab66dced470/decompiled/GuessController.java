import java.util.Scanner;

public class GuessController {
   private GuessEngine engine = new GuessEngine();
   private int attempts = 0;

   public String handleGuess(int value) {
      this.attempts++;
      return this.engine.compare(value);
   }

   public static void main(String[] args) {
      GuessController controller = new GuessController();
      Scanner sc = new Scanner(System.in);
      System.out.println("Guess a number between 1-100:");
      System.out.println(controller.handleGuess(sc.nextInt()));
   }
}
