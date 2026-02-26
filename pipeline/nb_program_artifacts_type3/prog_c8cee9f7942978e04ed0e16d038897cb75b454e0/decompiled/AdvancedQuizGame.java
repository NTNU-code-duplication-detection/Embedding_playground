public class AdvancedQuizGame {
   private String[] questions = new String[]{"2+2?", "Capital of France?"};
   private String[] answers = new String[]{"4", "Paris"};

   private boolean matches(int index, String input) {
      return this.answers[index].equalsIgnoreCase(input);
   }

   public int score(String[] responses) {
      int total = 0;

      for (int i = 0; i < this.questions.length; i++) {
         if (this.matches(i, responses[i])) {
            total++;
         }
      }

      return total;
   }

   public void ask(int index) {
      System.out.println(this.questions[index]);
   }

   public static void main(String[] args) {
      AdvancedQuizGame aq = new AdvancedQuizGame();
      aq.ask(0);
      System.out.println("Score: " + aq.score(new String[]{"4", "Paris"}));
   }
}
