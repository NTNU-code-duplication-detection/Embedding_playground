import java.util.ArrayList;
import java.util.List;

public class AdvancedToDo {
   private List<Task> tasks = new ArrayList<>();

   public void addTask(String var1) {
      this.tasks.add(new Task(var1));
   }

   public void removeTask(int var1) {
      this.tasks.remove(var1);
   }

   public void updateTask(int var1, String var2) {
      this.tasks.set(var1, new Task(var2));
   }

   public void viewTasks() {
      this.tasks.forEach(System.out::println);
   }

   public void markDone(int var1) {
      this.tasks.get(var1).markDone();
   }

   public static void main(String[] var0) {
      AdvancedToDo var1 = new AdvancedToDo();
      var1.addTask("Write code");
      var1.viewTasks();
   }
}
