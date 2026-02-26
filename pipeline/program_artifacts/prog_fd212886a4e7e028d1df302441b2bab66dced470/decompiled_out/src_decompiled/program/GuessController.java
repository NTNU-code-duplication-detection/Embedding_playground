import java.util.Scanner;

public class GuessController {
   private GuessEngine engine = new GuessEngine();
   private int attempts = 0;

   public String handleGuess(int var1) {
      this.attempts++;
      return this.engine.compare(var1);
   }

   public static void main(String[] var0) {
      GuessController var1 = new GuessController();
      Scanner var2 = new Scanner(System.in);
      System.out.println("Guess a number between 1-100:");
      System.out.println(var1.handleGuess(var2.nextInt()));
   }
}
