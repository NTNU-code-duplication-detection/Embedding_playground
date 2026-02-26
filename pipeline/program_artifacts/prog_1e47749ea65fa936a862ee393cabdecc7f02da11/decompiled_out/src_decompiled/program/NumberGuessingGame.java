import java.util.Random;
import java.util.Scanner;

public class NumberGuessingGame {
   private int number = new Random().nextInt(100) + 1;
   private int attempts = 0;

   public String guess(int var1) {
      this.attempts++;
      if (var1 == this.number) {
         return "Correct!";
      } else {
         return var1 < this.number ? "Too low" : "Too high";
      }
   }

   public int getAttempts() {
      return this.attempts;
   }

   public static void main(String[] var0) {
      NumberGuessingGame var1 = new NumberGuessingGame();
      Scanner var2 = new Scanner(System.in);
      System.out.println("Guess a number between 1-100:");
      int var3 = var2.nextInt();
      System.out.println(var1.guess(var3));
   }
}
