import java.util.Random;
import java.util.Scanner;

public class GuessSession {
   private int attempts = 0;
   private int number = new Random().nextInt(100) + 1;

   private String result(int guess) {
      if (guess == this.number) {
         return "Correct!";
      } else {
         return guess < this.number ? "Too low" : "Too high";
      }
   }

   public String submit(int guess) {
      this.attempts++;
      return this.result(guess);
   }

   public static void main(String[] args) {
      GuessSession session = new GuessSession();
      Scanner sc = new Scanner(System.in);
      System.out.println("Guess a number between 1-100:");
      System.out.println(session.submit(sc.nextInt()));
   }
}
