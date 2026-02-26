package googlejam2.p221;

import java.io.BufferedReader;

import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;


public class CounterCulture {

	/**
	 * @param args
	 */
	public static void  main(String[]  args) throws IOException {
		CounterCulture instance = new CounterCulture();
		BufferedReader br = new BufferedReader(new  FileReader("in.txt"));
		PrintWriter writer = new  PrintWriter("out.txt", "UTF-8");
		
		
		int MAX = 1000001;

		
		int[] res = new int[v1];

		
		v10[1] = 1;
		
		for(int  i=2;v4<v1;++v4) {
			int r = Integer.parseInt(new StringBuffer(new Integer(v4).toString()).reverse().toString());
			int r_r = Integer.parseInt(new StringBuffer(new Integer(v8).toString()).reverse().toString());
			if(v8<v4 && v4 == v9) {
				v10[v4] = Math.min(v10[v4-1], v10[v8])+1;
			}
			else
				v10[v4] = v10[v4-1]+1;
		}
		
		
		
		try {
		    String line = br.readLine();
		    
		    int t = Integer.parseInt(v6);
		     
		    for(int i=1;v4<=v11;++v4) {
		    	int n = Integer.parseInt(br.readLine());
		    	
		     	writer.print("Case #" + new Integer(v4).toString() + ": " + v10[v7]);
		     	
		    	if(v4<v11)
		    		writer.println();
		    }
		}  finally {
		      br.close();

		    writer.close();
		}
	}

}
