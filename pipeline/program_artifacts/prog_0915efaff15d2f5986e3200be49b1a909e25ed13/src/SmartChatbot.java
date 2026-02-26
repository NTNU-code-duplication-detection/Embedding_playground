public class SmartChatbot {

    private boolean contains(String text, String key) {
        return text.contains(key);
    }

    public String respond(String input) {
        if(contains(input, "hello")) return "Hi there!";
        if(contains(input, "how are you")) return "I'm fine!";
        if(contains(input, "bye")) return "Goodbye!";
        return "I don't understand.";
    }

    public static void main(String[] args) {
        SmartChatbot bot = new SmartChatbot();
        System.out.println(bot.respond("hello"));
    }
}
