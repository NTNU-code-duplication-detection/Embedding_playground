public class AccountManager {
   private double balance = 0.0;

   public void add(double var1) {
      this.balance += var1;
   }

   public void subtract(double var1) {
      this.balance -= var1;
   }

   public void transfer(AccountManager var1, double var2) {
      this.subtract(var2);
      var1.add(var2);
   }

   public void show() {
      System.out.println("Balance: " + this.balance);
   }

   public static void main(String[] var0) {
      AccountManager var1 = new AccountManager();
      AccountManager var2 = new AccountManager();
      var1.add(100.0);
      var1.transfer(var2, 50.0);
      var1.show();
      var2.show();
   }
}
