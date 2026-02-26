import java.util.ArrayList;
import java.util.List;

class TaskHelper {
    public static void markDone(List<String> list, int idx) {
        list.set(idx, list.get(idx) + " ✅");
    }
}

public class ToDoApp {
    private List<String> tasks = new ArrayList<>();

    public void add(String t) { tasks.add(t); }
    public void remove(int i) { tasks.remove(i); }
    public void update(int i, String t) { tasks.set(i, t); }
    public void show() { tasks.forEach(System.out::println); }
    public void complete(int i) { TaskHelper.markDone(tasks, i); }

    public static void main(String[] args) {
        ToDoApp app = new ToDoApp();
        app.add("Write code");
        app.show();
    }
}
