import java.util.List;

class TaskHelper {
   public static void markDone(List<String> var0, int var1) {
      var0.set(var1, (String)var0.get(var1) + " ✅");
   }
}
