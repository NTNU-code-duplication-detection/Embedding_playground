public class BankAccount {
    private double balance = 0;

    public void deposit(double amount) { balance += amount; }
    public void withdraw(double amount) { balance -= amount; }
    public double getBalance() { return balance; }
    public void transfer(BankAccount other, double amount) {
        this.withdraw(amount);
        other.deposit(amount);
    }
    public void printStatement() { System.out.println("Balance: " + balance); }

    public static void main(String[] args) {
        BankAccount a = new BankAccount();
        BankAccount b = new BankAccount();
        a.deposit(100);
        a.transfer(b, 50);
        a.printStatement();
        b.printStatement();
    }
}
