public class DialogueSession {
   public String handleInput(String text) {
      return this.match(text);
   }

   private String match(String input) {
      if (input.contains("hello")) {
         return "Hi there!";
      } else if (input.contains("how are you")) {
         return "I'm fine!";
      } else {
         return input.contains("bye") ? "Goodbye!" : "I don't understand.";
      }
   }

   public static void main(String[] args) {
      DialogueSession session = new DialogueSession();
      System.out.println(session.handleInput("hello"));
   }
}
