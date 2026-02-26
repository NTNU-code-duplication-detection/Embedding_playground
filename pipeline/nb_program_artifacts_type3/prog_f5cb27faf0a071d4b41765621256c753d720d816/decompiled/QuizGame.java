public class QuizGame {
   private String[] questions = new String[]{"2+2?", "Capital of France?"};
   private String[] answers = new String[]{"4", "Paris"};

   public boolean checkAnswer(int q, String a) {
      return this.answers[q].equalsIgnoreCase(a);
   }

   public void askQuestion(int q) {
      System.out.println(this.questions[q]);
   }

   public int score(String[] userAnswers) {
      int s = 0;

      for (int i = 0; i < this.questions.length; i++) {
         if (this.checkAnswer(i, userAnswers[i])) {
            s++;
         }
      }

      return s;
   }

   public static void main(String[] args) {
      QuizGame q = new QuizGame();
      q.askQuestion(0);
      System.out.println("Score: " + q.score(new String[]{"4", "Paris"}));
   }
}
