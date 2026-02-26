import java.util.ArrayList;
import java.util.List;

public class ToDoList {
    private List<String> tasks = new ArrayList<>();

    public void addTask(String task) { tasks.add(task); }
    public void removeTask(int index) { tasks.remove(index); }
    public void updateTask(int index, String task) { tasks.set(index, task); }
    public void viewTasks() { tasks.forEach(System.out::println); }
    public void markDone(int index) { tasks.set(index, tasks.get(index) + " ✅"); }

    public static void main(String[] args) {
        ToDoList list = new ToDoList();
        list.addTask("Write code");
        list.viewTasks();
    }
}
