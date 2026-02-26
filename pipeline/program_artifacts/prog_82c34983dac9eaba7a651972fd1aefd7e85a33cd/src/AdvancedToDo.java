import java.util.ArrayList;
import java.util.List;

class Task {
    private String description;
    private boolean done;

    public Task(String desc) {
        this.description = desc;
        this.done = false;
    }

    public void markDone() { this.done = true; }

    @Override
    public String toString() {
        return description + (done ? " ✅" : "");
    }
}

public class AdvancedToDo {
    private List<Task> tasks = new ArrayList<>();

    public void addTask(String desc) { tasks.add(new Task(desc)); }
    public void removeTask(int idx) { tasks.remove(idx); }
    public void updateTask(int idx, String desc) { tasks.set(idx, new Task(desc)); }
    public void viewTasks() { tasks.forEach(System.out::println); }
    public void markDone(int idx) { tasks.get(idx).markDone(); }

    public static void main(String[] args) {
        AdvancedToDo todo = new AdvancedToDo();
        todo.addTask("Write code");
        todo.viewTasks();
    }
}
