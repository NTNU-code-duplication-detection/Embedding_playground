public class SecureBankAccount {
   private double balance;

   private void update(double var1) {
      this.balance += var1;
   }

   public void deposit(double var1) {
      this.update(var1);
   }

   public void withdraw(double var1) {
      this.update(-var1);
   }

   public void transferTo(SecureBankAccount var1, double var2) {
      this.withdraw(var2);
      var1.deposit(var2);
   }

   public void printStatement() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] var0) {
      SecureBankAccount var1 = new SecureBankAccount();
      SecureBankAccount var2 = new SecureBankAccount();
      var1.deposit(100.0);
      var1.transferTo(var2, 50.0);
      var1.printStatement();
      var2.printStatement();
   }
}
