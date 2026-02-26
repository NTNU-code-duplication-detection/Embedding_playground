class ResponseEngine {

    public String evaluate(String msg) {
        if(msg.contains("hello")) return "Hi there!";
        if(msg.contains("how are you")) return "I'm fine!";
        if(msg.contains("bye")) return "Goodbye!";
        return "I don't understand.";
    }
}

public class ChatController {

    private ResponseEngine engine = new ResponseEngine();

    public String reply(String input) {
        return engine.evaluate(input);
    }

    public static void main(String[] args) {
        ChatController controller = new ChatController();
        System.out.println(controller.reply("hello"));
    }
}
