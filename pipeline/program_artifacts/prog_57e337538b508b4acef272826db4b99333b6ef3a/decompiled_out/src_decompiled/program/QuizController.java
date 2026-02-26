public class QuizController {
   private String[] questions = new String[]{"2+2?", "Capital of France?"};
   private AnswerChecker checker = new AnswerChecker();

   public void show(int var1) {
      System.out.println(this.questions[var1]);
   }

   public int score(String[] var1) {
      int var2 = 0;

      for (int var3 = 0; var3 < this.questions.length; var3++) {
         if (this.checker.correct(var3, var1[var3])) {
            var2++;
         }
      }

      return var2;
   }

   public static void main(String[] var0) {
      QuizController var1 = new QuizController();
      var1.show(0);
      System.out.println("Score: " + var1.score(new String[]{"4", "Paris"}));
   }
}
