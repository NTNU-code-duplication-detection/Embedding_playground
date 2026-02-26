public class QuizSession {
   private String[] q = new String[]{"2+2?", "Capital of France?"};
   private String[] a = new String[]{"4", "Paris"};

   public int run(String[] var1) {
      int var2 = 0;

      for (int var3 = 0; var3 < this.q.length; var3++) {
         if (this.a[var3].equalsIgnoreCase(var1[var3])) {
            var2++;
         }
      }

      return var2;
   }

   public void printQuestion(int var1) {
      System.out.println(this.q[var1]);
   }

   public static void main(String[] var0) {
      QuizSession var1 = new QuizSession();
      var1.printQuestion(0);
      System.out.println("Score: " + var1.run(new String[]{"4", "Paris"}));
   }
}
