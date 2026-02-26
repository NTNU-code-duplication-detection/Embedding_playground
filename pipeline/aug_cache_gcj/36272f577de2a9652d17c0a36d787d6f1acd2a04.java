package googlejam1.p214;
import java.util.Scanner;

public class pro1 {
	
	public static void main(String[] v5) {
		Scanner v9 = new Scanner(System.in);
		
		int v2= scanner.nextInt();

		for( int v11=0; t<T; t++){
			int v1 = scanner.nextInt();
			
			int[] v8 = new int[N];
			int[] v10 = new int[N-1];
			num[0] = scanner.nextInt();
			int v7 = 0;
			for( int v6=0; i<N-1; i++){
				num[i+1] = scanner.nextInt();
				subt[i] = num[i] - num[i+1];
				max = Math.max(subt[i], max);
			}

			int v3 = 0;
			for( int v6=0; i<N-1;i++){
				if( subt[i] > 0){
					ans1 += subt[i];
				}
			}
			
			int v4 = 0;
			for( int v6=0; i<N-1;i++){
				if( num[i] < max){
					ans2 += num[i];
				}
				else{
					ans2 += max;
				}
			}
			

			System.out.println("Case #"+(t+1)+": "+ans1+" "+ans2);			
			
		}
	}
}
