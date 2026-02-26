public class AccountManager {
   private double balance = 0.0;

   public void add(double amt) {
      this.balance += amt;
   }

   public void subtract(double amt) {
      this.balance -= amt;
   }

   public void transfer(AccountManager receiver, double amt) {
      this.subtract(amt);
      receiver.add(amt);
   }

   public void show() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] args) {
      AccountManager acc1 = new AccountManager();
      AccountManager acc2 = new AccountManager();
      acc1.add(100.0);
      acc1.transfer(acc2, 50.0);
      acc1.show();
      acc2.show();
   }
}
