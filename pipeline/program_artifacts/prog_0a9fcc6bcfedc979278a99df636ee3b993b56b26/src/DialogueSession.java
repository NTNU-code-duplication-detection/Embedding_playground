public class DialogueSession {

    public String handleInput(String text) {
        return match(text);
    }

    private String match(String input) {
        if(input.contains("hello")) return "Hi there!";
        if(input.contains("how are you")) return "I'm fine!";
        if(input.contains("bye")) return "Goodbye!";
        return "I don't understand.";
    }

    public static void main(String[] args) {
        DialogueSession session = new DialogueSession();
        System.out.println(session.handleInput("hello"));
    }
}
