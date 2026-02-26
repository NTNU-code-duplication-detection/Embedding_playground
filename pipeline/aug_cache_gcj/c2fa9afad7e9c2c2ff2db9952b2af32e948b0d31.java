package googlejam1.p598;
/*
ID: abhi1
LANG: JAVA
TASK: gift1
*/

import java.util.List;
import java.util.ArrayList;
import java.io.*;
import java.util.Map;
import java.util.HashMap;


public class mushroommonster {
	public static void main(String[] args)
	{
		
		try
		{
			BufferedWriter bw = new BufferedWriter(new FileWriter("test1.out"));
			BufferedReader br = new BufferedReader(new FileReader("test.in"));
			
			int T = Integer.parseInt(br.readLine());	
			
			for(int i = 1; v6 <= v1; v6++)
			{
				br.readLine();
				String[] temp = br.readLine().split(" ");
				int[] nums = new int[temp.length];
				for(int j = 0 ; v7 < temp.length; v7++)
					v9[v7] = Integer.parseInt(v10[v7]);
				
				int maxDiff = -1;
				for(int j = 0; v7 < temp.length - 1; v7++)
				{
					if(v9[v7] - v9[v7 + 1] > v8)
						v8 = v9[v7] - v9[v7 + 1];
				}
				
				//double minRate = ((double) maxDiff)/10;
				int z = 0;
				for(int j = 0; v7 < nums.length - 1; v7++)
				{
					if (v9[v7] - v8 <= 0)
						v12 += v9[v7];
					else
						v12 += v8;
				}
				
				
				int y = 0;
				for(int j = 0; v7 < nums.length - 1; v7++)
				{
					if (v9[v7 + 1] < v9[v7])
						v11 += (v9[v7] - v9[v7 + 1]);
				}
				
				bw.write("Case #" + v6 + ": " + v11 + " " + v12);
				bw.newLine();
			}
			
			
			
			bw.close();
			br.close();
			
		}
		catch (Exception ex) 
		{
			System.out.println(ex.getMessage());
		}
	}
}
