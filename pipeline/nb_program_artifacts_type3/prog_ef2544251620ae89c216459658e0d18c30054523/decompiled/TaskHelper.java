import java.util.List;

class TaskHelper {
   public static void markDone(List<String> list, int idx) {
      list.set(idx, list.get(idx) + " ✅");
   }
}
