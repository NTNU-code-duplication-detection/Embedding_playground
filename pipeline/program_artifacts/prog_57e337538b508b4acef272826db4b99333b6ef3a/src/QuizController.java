class AnswerChecker {
    private String[] answers = {"4", "Paris"};

    public boolean correct(int i, String a) {
        return answers[i].equalsIgnoreCase(a);
    }
}

public class QuizController {
    private String[] questions = {"2+2?", "Capital of France?"};
    private AnswerChecker checker = new AnswerChecker();

    public void show(int i) {
        System.out.println(questions[i]);
    }

    public int score(String[] responses) {
        int s = 0;
        for(int i = 0; i < questions.length; i++)
            if(checker.correct(i, responses[i])) s++;
        return s;
    }

    public static void main(String[] args) {
        QuizController qc = new QuizController();
        qc.show(0);
        System.out.println("Score: " + qc.score(new String[]{"4","Paris"}));
    }
}
