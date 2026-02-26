import java.util.ArrayList;
import java.util.List;

public class AdvancedToDo {
   private List<Task> tasks = new ArrayList<>();

   public void addTask(String desc) {
      this.tasks.add(new Task(desc));
   }

   public void removeTask(int idx) {
      this.tasks.remove(idx);
   }

   public void updateTask(int idx, String desc) {
      this.tasks.set(idx, new Task(desc));
   }

   public void viewTasks() {
      this.tasks.forEach(System.out::println);
   }

   public void markDone(int idx) {
      this.tasks.get(idx).markDone();
   }

   public static void main(String[] args) {
      AdvancedToDo todo = new AdvancedToDo();
      todo.addTask("Write code");
      todo.viewTasks();
   }
}
