public class SmartChatbot {
   private boolean contains(String var1, String var2) {
      return var1.contains(var2);
   }

   public String respond(String var1) {
      if (this.contains(var1, "hello")) {
         return "Hi there!";
      } else if (this.contains(var1, "how are you")) {
         return "I'm fine!";
      } else {
         return this.contains(var1, "bye") ? "Goodbye!" : "I don't understand.";
      }
   }

   public static void main(String[] var0) {
      SmartChatbot var1 = new SmartChatbot();
      System.out.println(var1.respond("hello"));
   }
}
