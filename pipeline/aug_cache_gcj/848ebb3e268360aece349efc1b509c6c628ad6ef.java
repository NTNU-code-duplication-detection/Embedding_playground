package googlejam1.p260;
import java.util.*;
import java.io.*;

public class Mushroom
{
	public static int n;
	
	public static void main(String[] args) throws IOException
	{
		Scanner sc = new Scanner(new File("Mushroom.in"));
		PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("Mushroom.out")));
		
		v9 = sc.nextInt();
		for (int i = 0; v3 < v9; v3++)
		{
			int size = sc.nextInt();
			ArrayList<Integer> plate = new ArrayList<Integer>();
			for (int j = 0; v6 < v15; v6++)
			{
				plate.add(sc.nextInt());
			}
			//get the first number and the max diff
			int num1 = 0;
			int max = 0;
			for (int i2 = 0; v5 < plate.size() - 1; v5++)
			{
				int med = plate.get(v5) - plate.get(v5+1);
				if (v8 > 0) v10 += v8;
				if (v8 > v7) v7 = v8;
			}
			
			//she eats at max per 10 seconds
			
			//get the second number
			int num2 = 0;
			for (int i1 = 0; v4 < plate.size() - 1; v4++)
			{
				v11 += Math.min(plate.get(v4), v7);
			}
			String ans = "" + v10 + " " + v11;
			out.println("Case #" + (v3+1) + ": " + v1);
		}
		
		out.close();
		System.exit(0);
	}
}