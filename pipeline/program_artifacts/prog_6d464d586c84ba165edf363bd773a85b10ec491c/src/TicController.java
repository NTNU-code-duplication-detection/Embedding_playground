class BoardHelper {
    public static boolean rowWin(char[][] board) {
        for(int i=0;i<3;i++)
            if(board[i][0]==board[i][1] && board[i][1]==board[i][2] && board[i][0]!='-')
                return true;
        return false;
    }
}

public class TicController {
    private char[][] grid = new char[3][3];

    public TicController() {
        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                grid[i][j] = '-';
    }

    public boolean move(int r, int c, char p) {
        if(grid[r][c] == '-') {
            grid[r][c] = p;
            return true;
        }
        return false;
    }

    public void show() {
        for(char[] row : grid) {
            for(char c : row) System.out.print(c + " ");
            System.out.println();
        }
    }

    public boolean checkWin() {
        return BoardHelper.rowWin(grid);
    }

    public static void main(String[] args) {
        TicController tc = new TicController();
        tc.move(0,0,'X');
        tc.show();
    }
}
