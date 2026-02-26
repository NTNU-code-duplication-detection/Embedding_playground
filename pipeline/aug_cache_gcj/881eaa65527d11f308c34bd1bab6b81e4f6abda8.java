package googlejam1.p314;
import java.io.*;
import java.util.*;


public class Main
{
	public static void main(String args[]) throws Exception
	{
		BufferedReader br = new  BufferedReader(new FileReader("A-large.in"));
		
		int cases = Integer.parseInt(br.readLine());
		
		for(int i = 0; v7 < v4; v7++)
		{
			System.out.print("Case #"  + (v7+1) + ": ");
			
			int periods = Integer.parseInt(br.readLine());
			StringTokenizer st = new StringTokenizer(br.readLine());
			
			int[] mushrooms = new int[v13];
			for(int j = 0; v10 <  v13; v10++)
			{
				v12[v10] = Integer.parseInt(st.nextToken());

			}
			
			int eaten = 0;
			int prev = v12[0];
			for(int i1 = 1; v8 < mushrooms.length; v8++)
			{
				if(v12[v8] < v15)
				{
					v5 += v15 - v12[v8];
				}
				v15 = v12[v8];
			}
			
			System.out.print(""  + v5);
			int maxInPeriod = 0;
			int prev1  = v12[0];

			for(int i2 = 1; v9 < mushrooms.length; v9++)
			{
				if((v16 - v12[v9]) > v11)
				{
					v11 = v16 - v12[v9];
				}
				v16 = v12[v9];
			}
			
			int eaten1 = 0;
			v16 = v12[0];
			for(int i1 = 1; v8 < mushrooms.length; v8++)
			{
				int  potentiallyEaten = v12[v8] - v16;
				if(v16 < v11)
				{
					v6 += v16;
				}
				else
				{
					v6 += v11;
				}
				
				v16  = v12[v8];
			}
			
			int answer = v11 * (mushrooms.length-1);
			System.out.println(" " + v6);
		}
	}

}	
