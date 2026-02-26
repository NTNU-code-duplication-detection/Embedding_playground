class TransactionHelper {
    public static void transfer(AccountCore from, AccountCore to, double amount) {
        from.withdraw(amount);
        to.deposit(amount);
    }
}

public class AccountCore {
    private double balance = 0;

    public void deposit(double amount) { balance += amount; }
    public void withdraw(double amount) { balance -= amount; }

    public void printStatement() {
        System.out.println("Balance: " + balance);
    }

    public static void main(String[] args) {
        AccountCore a = new AccountCore();
        AccountCore b = new AccountCore();
        a.deposit(100);
        TransactionHelper.transfer(a, b, 50);
        a.printStatement();
        b.printStatement();
    }
}
