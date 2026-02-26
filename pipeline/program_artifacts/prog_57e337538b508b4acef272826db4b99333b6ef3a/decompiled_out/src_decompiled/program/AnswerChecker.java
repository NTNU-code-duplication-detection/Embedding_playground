class AnswerChecker {
   private String[] answers = new String[]{"4", "Paris"};

   public boolean correct(int var1, String var2) {
      return this.answers[var1].equalsIgnoreCase(var2);
   }
}
