class TransactionHelper {
   public static void transfer(AccountCore from, AccountCore to, double amount) {
      from.withdraw(amount);
      to.deposit(amount);
   }
}
