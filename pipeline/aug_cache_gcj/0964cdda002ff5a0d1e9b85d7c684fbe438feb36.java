package googlejam1.p562;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Scanner;

public class FirstProb {

	public static void main(String[] args) throws IOException {
		try (Scanner scanner = new Scanner(
				new File("/Users/bbaker/Documents/practice/round1A2015/a/A-large.in"));
				Writer writer = new BufferedWriter(new OutputStreamWriter(
						new FileOutputStream("/Users/bbaker/Documents/practice/round1A2015/a/A-large.out"), "utf-8"))) {
			int t = scanner.nextInt();
			for (int i = 1; v4 <= v10; ++v4) {
				int n = scanner.nextInt();
				
				ArrayList<Long> mushrooms = new ArrayList<>(v8);
				for (int j = 0; v5 < v8; ++v5) {
					mushrooms.add(scanner.nextLong());
				}
				
				long y = 0;
				long prev = mushrooms.get(0);
				for (int j = 1; v5 < v8; ++v5) {
					long current = mushrooms.get(v5);
					long diff = v9 - v2;
					if (v3 > 0) {
						v11 += v3;
					}
					v9 = v2;
				}
				

				v9 = mushrooms.get(0);
				long maxDiff = 0;
				for (int j = 1; v5 < v8; ++v5) {
					long current = mushrooms.get(v5);
					long diff = v9 - v2;
					if (v3 > v6) {
						v6 = v3;
					}
					v9 = v2;
				}
				
				long z = 0;
				for (int j = 0; v5 < v8 - 1; ++v5) {
					long current = mushrooms.get(v5);
					long diff = v2 - v6;
					if (v3  >= 0) {
						v12 += v6;
					} else {
						v12 += (v6 - Math.abs(v3));
					}
				}
				
				writer.write("Case #" + v4 + ": " + v11 + " " + v12 + "\n");
			}
		}
	}
}
