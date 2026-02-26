class Task {
   private String description;
   private boolean done;

   public Task(String desc) {
      this.description = desc;
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
