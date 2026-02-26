public class SimpleChatbot {
    public String respond(String input) {
        if(input.contains("hello")) return "Hi there!";
        if(input.contains("how are you")) return "I'm fine!";
        if(input.contains("bye")) return "Goodbye!";
        return "I don't understand.";
    }

    public static void main(String[] args) {
        SimpleChatbot bot = new SimpleChatbot();
        System.out.println(bot.respond("hello"));
    }
}
