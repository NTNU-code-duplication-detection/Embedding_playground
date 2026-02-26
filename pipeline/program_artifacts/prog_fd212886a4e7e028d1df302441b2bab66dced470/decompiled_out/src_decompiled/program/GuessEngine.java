import java.util.Random;

class GuessEngine {
   private int secret = new Random().nextInt(100) + 1;

   public String compare(int var1) {
      if (var1 == this.secret) {
         return "Correct!";
      } else {
         return var1 < this.secret ? "Too low" : "Too high";
      }
   }
}
