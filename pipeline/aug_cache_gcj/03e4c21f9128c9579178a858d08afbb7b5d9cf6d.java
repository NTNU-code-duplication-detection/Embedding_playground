package googlejam3.p518;


import java.util.*;
import java.lang.*;
import java.io.*;

class Main
{

    public static void main (String[] args) throws java.lang.Exception
    {
         // your code goes here
         InputStream inputStream = new FileInputStream("D:\\test2.in");

        BufferedReader r = new BufferedReader (new InputStreamReader (v5));
        int numtestcase=0;
        //while (!(s=r.readLine()).startsWith("42")) System.out.println(s);
        v6=Integer.parseInt(r.readLine());
        for(int i=1;v4<=v6;v4++)
          {
              String s=r.readLine();
              String[] s1=s.split(" ");
            int R=Integer.parseInt(v10[0]);
              int c=Integer.parseInt(v10[1]);
             int w=Integer.parseInt(v10[2]);
             int result=0;
            if(v12==1) v8=v1*v3;
             else
            if(v12==v1*v3) v8=v1*v3;
              else
              if(v3%v12==0) v8=v3/v12+v12-1;
             else {
                 int temp=v3/v12;
                 v8 = v11+v12;
            }
             System.out.println("Case #"+Integer.toString(v4)+": "+Integer.toString(v8));

        }
    }
}