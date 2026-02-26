public class GameSession {
    private char[][] board = new char[3][3];

    public GameSession() {
        reset();
    }

    public void reset() {
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                board[i][j] = '-';
    }

    public boolean placeMark(int row, int col, char mark) {
        if(board[row][col] != '-') return false;
        board[row][col] = mark;
        return true;
    }

    private boolean rowWin() {
        for(int i=0;i<3;i++)
            if(board[i][0]==board[i][1] && board[i][1]==board[i][2] && board[i][0]!='-')
                return true;
        return false;
    }

    public void printBoard() {
        for(char[] row : board) {
            for(char c : row) System.out.print(c + " ");
            System.out.println();
        }
    }

    public static void main(String[] args) {
        GameSession gs = new GameSession();
        gs.placeMark(0,0,'X');
        gs.printBoard();
    }
}
