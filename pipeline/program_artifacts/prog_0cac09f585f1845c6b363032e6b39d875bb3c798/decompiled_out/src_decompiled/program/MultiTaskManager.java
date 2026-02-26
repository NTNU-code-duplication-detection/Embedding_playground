import java.util.ArrayList;
import java.util.List;

public class MultiTaskManager {
   private List<String> taskList = new ArrayList<>();

   private String done(String var1) {
      return var1 + " ✅";
   }

   public void addTask(String var1) {
      this.taskList.add(var1);
   }

   public void removeTask(int var1) {
      this.taskList.remove(var1);
   }

   public void updateTask(int var1, String var2) {
      this.taskList.set(var1, var2);
   }

   public void markTaskDone(int var1) {
      this.taskList.set(var1, this.done(this.taskList.get(var1)));
   }

   public void displayTasks() {
      for (String var2 : this.taskList) {
         System.out.println(var2);
      }
   }

   public static void main(String[] var0) {
      MultiTaskManager var1 = new MultiTaskManager();
      var1.addTask("Write code");
      var1.displayTasks();
   }
}
