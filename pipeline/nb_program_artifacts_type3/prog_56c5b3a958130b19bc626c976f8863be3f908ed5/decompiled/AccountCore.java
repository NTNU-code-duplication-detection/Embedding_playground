public class AccountCore {
   private double balance = 0.0;

   public void deposit(double amount) {
      this.balance += amount;
   }

   public void withdraw(double amount) {
      this.balance -= amount;
   }

   public void printStatement() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] args) {
      AccountCore a = new AccountCore();
      AccountCore b = new AccountCore();
      a.deposit(100.0);
      TransactionHelper.transfer(a, b, 50.0);
      a.printStatement();
      b.printStatement();
   }
}
