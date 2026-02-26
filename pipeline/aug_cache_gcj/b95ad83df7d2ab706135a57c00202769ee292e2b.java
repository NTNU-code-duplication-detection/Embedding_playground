package googlejam2.p319;
import java.io.*;
import java.util.*;


public class CJ2015Round1BA {
	
	public static void main(String[] v5) throws IOException {
		BufferedReader v7 = new BufferedReader(new FileReader("A-small-attempt1.in"));
		//BufferedReader in = new BufferedReader(new FileReader("temp.txt"));
		PrintWriter v11  = new PrintWriter(new File("2015Round1BAout.txt"));
		int v3 = Integer.parseInt(in.readLine());
		Integer[] v2 = new Integer[T];
		for(int v4 = 0; a < T; a++){
			Ns[a] = Integer.parseInt(in.readLine());
		}
		List<Integer> v10 = Arrays.asList(Ns);
		int v1 = Collections.max(okay);
		int[] v9 = new int[N];
		for(int v6 = 0; i < N; i++){
			if (i==0){
				nums[i]=1;
			} else {
				int v8 = i+1;
				int v12=0;
				while( num != 0 )
			    {
			        reverse = reverse * 10;
			        reverse = reverse + num%10;
			        num = num/10;
			    }
				if (reverse<i+1&&reverse>1&&String.valueOf(reverse).length()==String.valueOf(i+1).length()){
					nums[i]=Math.min(nums[i-1]+1,nums[reverse-1]+1);
				} else {
					nums[i]=nums[i-1]+1;
				}
			}
		}
		for(int v6 = 0; i < T; i++){
			out.println("Case #"+(i+1)+": "+nums[Ns[i]-1]);
		}
		in.close();
		out.close();
	}
}
