public class SimpleChatbot {
   public String respond(String input) {
      if (input.contains("hello")) {
         return "Hi there!";
      } else if (input.contains("how are you")) {
         return "I'm fine!";
      } else {
         return input.contains("bye") ? "Goodbye!" : "I don't understand.";
      }
   }

   public static void main(String[] args) {
      SimpleChatbot bot = new SimpleChatbot();
      System.out.println(bot.respond("hello"));
   }
}
