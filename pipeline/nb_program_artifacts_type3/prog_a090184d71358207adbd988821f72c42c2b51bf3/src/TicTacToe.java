public class TicTacToe {
    private char[][] board = new char[3][3];

    public TicTacToe() {
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) board[i][j] = '-';
    }

    public boolean makeMove(int row, int col, char player) {
        if(board[row][col] == '-') { board[row][col] = player; return true; }
        return false;
    }

    public void printBoard() {
        for(char[] row: board) { for(char c: row) System.out.print(c+" "); System.out.println(); }
    }

    public boolean checkWin() {
        // Simple horizontal check for demo
        for(int i=0;i<3;i++) if(board[i][0]==board[i][1] && board[i][1]==board[i][2] && board[i][0]!='-') return true;
        return false;
    }

    public void reset() { for(int i=0;i<3;i++) for(int j=0;j<3;j++) board[i][j]='-'; }

    public static void main(String[] args) {
        TicTacToe game = new TicTacToe();
        game.makeMove(0,0,'X');
        game.printBoard();
    }
}
