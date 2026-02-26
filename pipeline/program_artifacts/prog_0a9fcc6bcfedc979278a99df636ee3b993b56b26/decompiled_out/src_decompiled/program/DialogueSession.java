public class DialogueSession {
   public String handleInput(String var1) {
      return this.match(var1);
   }

   private String match(String var1) {
      if (var1.contains("hello")) {
         return "Hi there!";
      } else if (var1.contains("how are you")) {
         return "I'm fine!";
      } else {
         return var1.contains("bye") ? "Goodbye!" : "I don't understand.";
      }
   }

   public static void main(String[] var0) {
      DialogueSession var1 = new DialogueSession();
      System.out.println(var1.handleInput("hello"));
   }
}
