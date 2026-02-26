import java.util.Random;
import java.util.Scanner;

public class NumberGuessingGame {
   private int number = new Random().nextInt(100) + 1;
   private int attempts = 0;

   public String guess(int n) {
      this.attempts++;
      if (n == this.number) {
         return "Correct!";
      } else {
         return n < this.number ? "Too low" : "Too high";
      }
   }

   public int getAttempts() {
      return this.attempts;
   }

   public static void main(String[] args) {
      NumberGuessingGame game = new NumberGuessingGame();
      Scanner sc = new Scanner(System.in);
      System.out.println("Guess a number between 1-100:");
      int input = sc.nextInt();
      System.out.println(game.guess(input));
   }
}
