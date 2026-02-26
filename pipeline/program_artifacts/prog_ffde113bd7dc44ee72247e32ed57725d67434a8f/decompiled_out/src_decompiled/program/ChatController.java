public class ChatController {
   private ResponseEngine engine = new ResponseEngine();

   public String reply(String var1) {
      return this.engine.evaluate(var1);
   }

   public static void main(String[] var0) {
      ChatController var1 = new ChatController();
      System.out.println(var1.reply("hello"));
   }
}
