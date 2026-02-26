public class AccountManager {
    private double balance = 0;

    public void add(double amt) { balance += amt; }
    public void subtract(double amt) { balance -= amt; }

    public void transfer(AccountManager receiver, double amt) {
        subtract(amt);
        receiver.add(amt);
    }

    public void show() {
        System.out.println("Balance: " + balance);
    }

    public static void main(String[] args) {
        AccountManager acc1 = new AccountManager();
        AccountManager acc2 = new AccountManager();
        acc1.add(100);
        acc1.transfer(acc2, 50);
        acc1.show();
        acc2.show();
    }
}
