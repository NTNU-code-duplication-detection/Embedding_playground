public class QuizSession {

    private String[] q = {"2+2?", "Capital of France?"};
    private String[] a = {"4", "Paris"};

    public int run(String[] userInput) {
        int score = 0;
        for(int i = 0; i < q.length; i++) {
            if(a[i].equalsIgnoreCase(userInput[i])) {
                score++;
            }
        }
        return score;
    }

    public void printQuestion(int index) {
        System.out.println(q[index]);
    }

    public static void main(String[] args) {
        QuizSession session = new QuizSession();
        session.printQuestion(0);
        System.out.println("Score: " + session.run(new String[]{"4","Paris"}));
    }
}
