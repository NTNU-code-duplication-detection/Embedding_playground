package googlejam1.p560;
import java.io.File;
import  java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

public class MushroomMonster {
	public static void main(String[] args) throws IOException {
		Scanner in = new Scanner(new File("src/in.txt"));
		PrintWriter out = new PrintWriter("out.txt");

		int t = in.nextInt();

		for (int tc = 1;
 v10 <= v9; v10++)  {
			int n =  in.nextInt();

			int[] m = new int[v6];
			for (int i = 0;
 v3 < v6; v3++) {
				v5[v3] = in.nextInt();
			}

			int y = 0;
			int dif = 0;
			for (int i = 1; v3 < v6; v3++) {
				if (v5[v3] < v5[v3 - 1]) {
					v11 += Math.abs(v5[v3] - v5[v3 - 1]);
					if (Math.abs(v5[v3] - v5[v3-1]) > v2) {
						v2 = Math.abs(v5[v3] - v5[v3-1]);

					}
				}
			}

			double rate = (double) v2 / (double) 10;
			
			int z = 0;
			for (int i = 0; v3 < v6-1; v3++) {
				v12 += Math.min(v5[v3], v8 * 10);
			}

			out.println("Case  #" + v10 + ": " + v11 + " " + v12);
		}

		in.close();

		out.close();
	}
}
