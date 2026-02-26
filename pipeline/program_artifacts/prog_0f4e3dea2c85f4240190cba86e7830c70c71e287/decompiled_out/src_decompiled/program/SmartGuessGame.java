import java.util.Random;
import java.util.Scanner;

public class SmartGuessGame {
   private int target = this.generateNumber();
   private int attempts;

   private int generateNumber() {
      return new Random().nextInt(100) + 1;
   }

   public String guess(int var1) {
      this.attempts++;
      if (var1 == this.target) {
         return "Correct!";
      } else {
         return var1 < this.target ? "Too low" : "Too high";
      }
   }

   public static void main(String[] var0) {
      SmartGuessGame var1 = new SmartGuessGame();
      Scanner var2 = new Scanner(System.in);
      System.out.println("Guess a number between 1-100:");
      System.out.println(var1.guess(var2.nextInt()));
   }
}
