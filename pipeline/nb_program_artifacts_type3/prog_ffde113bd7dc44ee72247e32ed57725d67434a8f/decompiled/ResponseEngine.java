class ResponseEngine {
   public String evaluate(String msg) {
      if (msg.contains("hello")) {
         return "Hi there!";
      } else if (msg.contains("how are you")) {
         return "I'm fine!";
      } else {
         return msg.contains("bye") ? "Goodbye!" : "I don't understand.";
      }
   }
}
