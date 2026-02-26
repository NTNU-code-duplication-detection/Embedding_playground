class AnswerChecker {
   private String[] answers = new String[]{"4", "Paris"};

   public boolean correct(int i, String a) {
      return this.answers[i].equalsIgnoreCase(a);
   }
}
