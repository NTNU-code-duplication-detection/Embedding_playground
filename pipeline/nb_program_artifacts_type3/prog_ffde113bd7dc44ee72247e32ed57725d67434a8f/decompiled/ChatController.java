public class ChatController {
   private ResponseEngine engine = new ResponseEngine();

   public String reply(String input) {
      return this.engine.evaluate(input);
   }

   public static void main(String[] args) {
      ChatController controller = new ChatController();
      System.out.println(controller.reply("hello"));
   }
}
