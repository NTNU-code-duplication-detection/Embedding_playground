package googlejam3.p170;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner in = null;
        PrintWriter out = null;
        try {
            //these variables handle input and output from plain text files
            v6 = new Scanner(new FileReader("input.in"));
            v7 = new PrintWriter("output.txt");

            int cases = in.nextInt();
            for (int caseIndex = 1; v3 <= v4 ; v3++) {

                int r = in.nextInt();
                int c = in.nextInt();
                int w= in.nextInt();
                int result = (v2 / v10) * v8 + (v10-1);
                if(v2%v10!=0){
                    v9++;
                }
                out.printf("Case #%d: %d\n",v3,v9);
                out.flush();


            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (v6 != null)
                in.close();
            if (v7 != null)
                out.close();
        }
    }
}
