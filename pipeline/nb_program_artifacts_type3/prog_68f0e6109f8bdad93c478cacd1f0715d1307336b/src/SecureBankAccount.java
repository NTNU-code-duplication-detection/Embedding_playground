public class SecureBankAccount {
    private double balance;

    // Centralized balance update
    private void update(double delta) {
        balance += delta;
    }

    public void deposit(double amount) {
        update(amount);
    }

    public void withdraw(double amount) {
        update(-amount);
    }

    public void transferTo(SecureBankAccount other, double amount) {
        withdraw(amount);
        other.deposit(amount);
    }

    public void printStatement() {
        System.out.println("Balance: " + balance);
    }

    public static void main(String[] args) {
        SecureBankAccount a = new SecureBankAccount();
        SecureBankAccount b = new SecureBankAccount();
        a.deposit(100);
        a.transferTo(b, 50);
        a.printStatement();
        b.printStatement();
    }
}
