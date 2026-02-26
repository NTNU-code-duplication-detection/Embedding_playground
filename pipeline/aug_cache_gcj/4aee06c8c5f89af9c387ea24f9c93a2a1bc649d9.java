package googlejam2.p293;
import java.util.*;
import java.io.*;

public class CodeJamCounter{
    public static int[] v3;

    public static void main(String[] v1) throws Exception{
        count = new int[1000001];
        count[1] = 1;
        BufferedReader v2 = new BufferedReader(new InputStreamReader(System.in));
        StringBuilder v7 = new StringBuilder();
        int v10 = Integer.parseInt(br.readLine().trim());
        for(int v4 = 1; i < 1000000; i++){
            StringBuilder v9 = new StringBuilder();
            sb.append(i);
            int v8 = Integer.parseInt(sb.reverse().toString());
            if(count[i+1] == 0) count[i+1] = count[i] + 1;
            else if(count[i] + 1 < count[i+1]) count[i+1] = count[i] + 1;
            if(count[rev] == 0) count[rev] = count[i] + 1;
            else if(count[i] + 1 < count[rev]) count[rev] = count[i] + 1;
        }
        for(int v5 = 1; k <= t; k++){
            int v6 = Integer.parseInt(br.readLine().trim());
            out.append("Case #" + k + ": " + count[n] + "\n");
        }
        System.out.print(out);
    }
}