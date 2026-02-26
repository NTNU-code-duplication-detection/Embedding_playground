class Task {
   private String description;
   private boolean done;

   public Task(String var1) {
      this.description = var1;
      this.done = false;
   }

   public void markDone() {
      this.done = true;
   }

   @Override
   public String toString() {
      return this.description + (this.done ? " ✅" : "");
   }
}
