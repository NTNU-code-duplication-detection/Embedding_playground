import java.util.ArrayList;
import java.util.List;

public class MultiTaskManager {
    private List<String> taskList;

    public MultiTaskManager() { taskList = new ArrayList<>(); }

    // helper function to append checkmark
    private String done(String task) { return task + " ✅"; }

    public void addTask(String t) { taskList.add(t); }
    public void removeTask(int idx) { taskList.remove(idx); }
    public void updateTask(int idx, String t) { taskList.set(idx, t); }
    public void markTaskDone(int idx) { taskList.set(idx, done(taskList.get(idx))); }

    public void displayTasks() { 
        for (String t : taskList) System.out.println(t); 
    }

    public static void main(String[] args) {
        MultiTaskManager manager = new MultiTaskManager();
        manager.addTask("Write code");
        manager.displayTasks();
    }
}
