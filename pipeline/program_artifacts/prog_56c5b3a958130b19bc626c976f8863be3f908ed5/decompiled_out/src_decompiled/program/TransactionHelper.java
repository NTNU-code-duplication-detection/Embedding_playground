class TransactionHelper {
   public static void transfer(AccountCore var0, AccountCore var1, double var2) {
      var0.withdraw(var2);
      var1.deposit(var2);
   }
}
