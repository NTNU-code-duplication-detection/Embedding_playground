public class BankAccount {
   private double balance = 0.0;

   public void deposit(double var1) {
      this.balance += var1;
   }

   public void withdraw(double var1) {
      this.balance -= var1;
   }

   public double getBalance() {
      return this.balance;
   }

   public void transfer(BankAccount var1, double var2) {
      this.withdraw(var2);
      var1.deposit(var2);
   }

   public void printStatement() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] var0) {
      BankAccount var1 = new BankAccount();
      BankAccount var2 = new BankAccount();
      var1.deposit(100.0);
      var1.transfer(var2, 50.0);
      var1.printStatement();
      var2.printStatement();
   }
}
