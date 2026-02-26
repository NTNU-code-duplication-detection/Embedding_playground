public class AdvancedTicTacToe {
    private char[][] board;

    public AdvancedTicTacToe() {
        board = createBoard();
    }

    // Helper to initialize board
    private char[][] createBoard() {
        char[][] b = new char[3][3];
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                b[i][j] = '-';
        return b;
    }

    public boolean makeMove(int row, int col, char player) {
        if(board[row][col] != '-') return false;
        board[row][col] = player;
        return true;
    }

    public boolean checkHorizontalWin() {
        for(int i=0;i<3;i++)
            if(board[i][0]==board[i][1] && board[i][1]==board[i][2] && board[i][0]!='-')
                return true;
        return false;
    }

    public void display() {
        for(char[] r : board) {
            for(char c : r) System.out.print(c + " ");
            System.out.println();
        }
    }

    public static void main(String[] args) {
        AdvancedTicTacToe game = new AdvancedTicTacToe();
        game.makeMove(0,0,'X');
        game.display();
    }
}
