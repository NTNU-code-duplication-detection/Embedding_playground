import java.util.ArrayList;
import java.util.List;

public class MultiTaskManager {
   private List<String> taskList = new ArrayList<>();

   private String done(String task) {
      return task + " ✅";
   }

   public void addTask(String t) {
      this.taskList.add(t);
   }

   public void removeTask(int idx) {
      this.taskList.remove(idx);
   }

   public void updateTask(int idx, String t) {
      this.taskList.set(idx, t);
   }

   public void markTaskDone(int idx) {
      this.taskList.set(idx, this.done(this.taskList.get(idx)));
   }

   public void displayTasks() {
      for (String t : this.taskList) {
         System.out.println(t);
      }
   }

   public static void main(String[] args) {
      MultiTaskManager manager = new MultiTaskManager();
      manager.addTask("Write code");
      manager.displayTasks();
   }
}
