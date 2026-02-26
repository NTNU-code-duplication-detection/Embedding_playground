public class SimpleChatbot {
   public String respond(String var1) {
      if (var1.contains("hello")) {
         return "Hi there!";
      } else if (var1.contains("how are you")) {
         return "I'm fine!";
      } else {
         return var1.contains("bye") ? "Goodbye!" : "I don't understand.";
      }
   }

   public static void main(String[] var0) {
      SimpleChatbot var1 = new SimpleChatbot();
      System.out.println(var1.respond("hello"));
   }
}
