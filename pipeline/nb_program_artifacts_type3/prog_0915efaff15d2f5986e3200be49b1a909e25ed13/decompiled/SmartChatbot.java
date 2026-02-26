public class SmartChatbot {
   private boolean contains(String text, String key) {
      return text.contains(key);
   }

   public String respond(String input) {
      if (this.contains(input, "hello")) {
         return "Hi there!";
      } else if (this.contains(input, "how are you")) {
         return "I'm fine!";
      } else {
         return this.contains(input, "bye") ? "Goodbye!" : "I don't understand.";
      }
   }

   public static void main(String[] args) {
      SmartChatbot bot = new SmartChatbot();
      System.out.println(bot.respond("hello"));
   }
}
