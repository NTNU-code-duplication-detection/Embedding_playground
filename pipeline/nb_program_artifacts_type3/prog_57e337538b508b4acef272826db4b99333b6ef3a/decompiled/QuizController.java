public class QuizController {
   private String[] questions = new String[]{"2+2?", "Capital of France?"};
   private AnswerChecker checker = new AnswerChecker();

   public void show(int i) {
      System.out.println(this.questions[i]);
   }

   public int score(String[] responses) {
      int s = 0;

      for (int i = 0; i < this.questions.length; i++) {
         if (this.checker.correct(i, responses[i])) {
            s++;
         }
      }

      return s;
   }

   public static void main(String[] args) {
      QuizController qc = new QuizController();
      qc.show(0);
      System.out.println("Score: " + qc.score(new String[]{"4", "Paris"}));
   }
}
