import java.util.Random;

class GuessEngine {
   private int secret = new Random().nextInt(100) + 1;

   public String compare(int input) {
      if (input == this.secret) {
         return "Correct!";
      } else {
         return input < this.secret ? "Too low" : "Too high";
      }
   }
}
