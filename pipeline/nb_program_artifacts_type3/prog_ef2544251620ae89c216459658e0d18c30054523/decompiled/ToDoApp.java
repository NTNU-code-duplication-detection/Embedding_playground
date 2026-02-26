import java.util.ArrayList;
import java.util.List;

public class ToDoApp {
   private List<String> tasks = new ArrayList<>();

   public void add(String t) {
      this.tasks.add(t);
   }

   public void remove(int i) {
      this.tasks.remove(i);
   }

   public void update(int i, String t) {
      this.tasks.set(i, t);
   }

   public void show() {
      this.tasks.forEach(System.out::println);
   }

   public void complete(int i) {
      TaskHelper.markDone(this.tasks, i);
   }

   public static void main(String[] args) {
      ToDoApp app = new ToDoApp();
      app.add("Write code");
      app.show();
   }
}
