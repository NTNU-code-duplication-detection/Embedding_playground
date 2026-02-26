public class AccountCore {
   private double balance = 0.0;

   public void deposit(double var1) {
      this.balance += var1;
   }

   public void withdraw(double var1) {
      this.balance -= var1;
   }

   public void printStatement() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] var0) {
      AccountCore var1 = new AccountCore();
      AccountCore var2 = new AccountCore();
      var1.deposit(100.0);
      TransactionHelper.transfer(var1, var2, 50.0);
      var1.printStatement();
      var2.printStatement();
   }
}
