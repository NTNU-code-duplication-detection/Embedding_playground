import java.util.ArrayList;
import java.util.List;

public class ToDoApp {
   private List<String> tasks = new ArrayList<>();

   public void add(String var1) {
      this.tasks.add(var1);
   }

   public void remove(int var1) {
      this.tasks.remove(var1);
   }

   public void update(int var1, String var2) {
      this.tasks.set(var1, var2);
   }

   public void show() {
      this.tasks.forEach(System.out::println);
   }

   public void complete(int var1) {
      TaskHelper.markDone(this.tasks, var1);
   }

   public static void main(String[] var0) {
      ToDoApp var1 = new ToDoApp();
      var1.add("Write code");
      var1.show();
   }
}
