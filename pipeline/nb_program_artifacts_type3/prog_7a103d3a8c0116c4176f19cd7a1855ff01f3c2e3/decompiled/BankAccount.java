public class BankAccount {
   private double balance = 0.0;

   public void deposit(double amount) {
      this.balance += amount;
   }

   public void withdraw(double amount) {
      this.balance -= amount;
   }

   public double getBalance() {
      return this.balance;
   }

   public void transfer(BankAccount other, double amount) {
      this.withdraw(amount);
      other.deposit(amount);
   }

   public void printStatement() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] args) {
      BankAccount a = new BankAccount();
      BankAccount b = new BankAccount();
      a.deposit(100.0);
      a.transfer(b, 50.0);
      a.printStatement();
      b.printStatement();
   }
}
