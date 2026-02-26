import java.util.ArrayList;
import java.util.List;

public class ToDoList {
   private List<String> tasks = new ArrayList<>();

   public void addTask(String task) {
      this.tasks.add(task);
   }

   public void removeTask(int index) {
      this.tasks.remove(index);
   }

   public void updateTask(int index, String task) {
      this.tasks.set(index, task);
   }

   public void viewTasks() {
      this.tasks.forEach(System.out::println);
   }

   public void markDone(int index) {
      this.tasks.set(index, this.tasks.get(index) + " ✅");
   }

   public static void main(String[] args) {
      ToDoList list = new ToDoList();
      list.addTask("Write code");
      list.viewTasks();
   }
}
