class ResponseEngine {
   public String evaluate(String var1) {
      if (var1.contains("hello")) {
         return "Hi there!";
      } else if (var1.contains("how are you")) {
         return "I'm fine!";
      } else {
         return var1.contains("bye") ? "Goodbye!" : "I don't understand.";
      }
   }
}
