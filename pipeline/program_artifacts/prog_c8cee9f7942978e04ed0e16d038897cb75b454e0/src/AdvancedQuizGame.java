public class AdvancedQuizGame {

    private String[] questions = {"2+2?", "Capital of France?"};
    private String[] answers = {"4", "Paris"};

    // Helper to verify answer
    private boolean matches(int index, String input) {
        return answers[index].equalsIgnoreCase(input);
    }

    public int score(String[] responses) {
        int total = 0;
        for(int i = 0; i < questions.length; i++)
            if(matches(i, responses[i])) total++;
        return total;
    }

    public void ask(int index) {
        System.out.println(questions[index]);
    }

    public static void main(String[] args) {
        AdvancedQuizGame aq = new AdvancedQuizGame();
        aq.ask(0);
        System.out.println("Score: " + aq.score(new String[]{"4","Paris"}));
    }
}
