public class SecureBankAccount {
   private double balance;

   private void update(double delta) {
      this.balance += delta;
   }

   public void deposit(double amount) {
      this.update(amount);
   }

   public void withdraw(double amount) {
      this.update(-amount);
   }

   public void transferTo(SecureBankAccount other, double amount) {
      this.withdraw(amount);
      other.deposit(amount);
   }

   public void printStatement() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] args) {
      SecureBankAccount a = new SecureBankAccount();
      SecureBankAccount b = new SecureBankAccount();
      a.deposit(100.0);
      a.transferTo(b, 50.0);
      a.printStatement();
      b.printStatement();
   }
}
