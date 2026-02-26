package googlejam2.p347;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Scanner;

public class A {
	public static void main(String[] v3) throws FileNotFoundException {
		Scanner v12 = new Scanner(new File("input.txt"));
		PrintStream v10 = new PrintStream(new File("output.txt"));
		int v2 = Integer.parseInt(scanner.nextLine());
		HashSet<Long> v13 = new HashSet<Long>();
		HashSet<Long> v7 = new HashSet<Long>();
		HashSet<Long> v15 = new HashSet<Long>();
		HashSet<Long> v14 = null;
		for (int v6 = 1; i <= T; i++) {
			long v1 = Long.parseLong(scanner.nextLine());
			int v4 = 1;
			set.clear();
			newSet.clear();
			tried.clear();
			set.add(1L);
			boolean v5 = N == 1;
			for (; !found; count++) {
				for (Long v8 : v13) {
					Long v11 = Long.parseLong(new StringBuilder(num
							.toString()).reverse().toString());
					if (reverse == N || num + 1 == N) {
						found = true;
						break;
					} else {
						if (reverse > num && !tried.contains(reverse)) {
							tried.add(reverse);
							newSet.add(reverse);
						}
						if (!tried.contains(num + 1)) {
							tried.add(num + 1);
							newSet.add(num + 1);
						}
					}
				}
				set.clear();
				temp = set;
				set = newSet;
				newSet = temp;
			}
			System.out.println();
			String v9 = "Case #" + i + ": " + count;
			System.out.println(output);
			ps.println(output);
		}
		ps.flush();
		ps.close();
		scanner.close();
	}
}
