public class AdvancedQuizGame {
   private String[] questions = new String[]{"2+2?", "Capital of France?"};
   private String[] answers = new String[]{"4", "Paris"};

   private boolean matches(int var1, String var2) {
      return this.answers[var1].equalsIgnoreCase(var2);
   }

   public int score(String[] var1) {
      int var2 = 0;

      for (int var3 = 0; var3 < this.questions.length; var3++) {
         if (this.matches(var3, var1[var3])) {
            var2++;
         }
      }

      return var2;
   }

   public void ask(int var1) {
      System.out.println(this.questions[var1]);
   }

   public static void main(String[] var0) {
      AdvancedQuizGame var1 = new AdvancedQuizGame();
      var1.ask(0);
      System.out.println("Score: " + var1.score(new String[]{"4", "Paris"}));
   }
}
