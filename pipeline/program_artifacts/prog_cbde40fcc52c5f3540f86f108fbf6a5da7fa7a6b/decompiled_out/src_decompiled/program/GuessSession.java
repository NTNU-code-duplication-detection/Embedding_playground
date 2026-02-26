import java.util.Random;
import java.util.Scanner;

public class GuessSession {
   private int attempts = 0;
   private int number = new Random().nextInt(100) + 1;

   private String result(int var1) {
      if (var1 == this.number) {
         return "Correct!";
      } else {
         return var1 < this.number ? "Too low" : "Too high";
      }
   }

   public String submit(int var1) {
      this.attempts++;
      return this.result(var1);
   }

   public static void main(String[] var0) {
      GuessSession var1 = new GuessSession();
      Scanner var2 = new Scanner(System.in);
      System.out.println("Guess a number between 1-100:");
      System.out.println(var1.submit(var2.nextInt()));
   }
}
