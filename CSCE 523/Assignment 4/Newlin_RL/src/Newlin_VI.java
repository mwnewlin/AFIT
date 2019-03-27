//package org.jfree.chart.demo;
import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
/*
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
*/

public class Newlin_VI {

    public static final double INF = 10000.0;
    private static int length;
    private static int width;
    private static double[][] vs;
    private static double[][] vs_prime;
    private static double[][] es;
    private static double p_forward = 0.9;
    private static double p_stay = 0.07;
    private static double p_backward = 0.03;
    private static String[][] maze;
    private static int startX;
    private static int startY;
    private static double move_cost = 1.0;
    static ArrayList<Double> valueIterationRewards;

    public static double getReward(String s) {
        if (s.equals("G")) {
            return (10-move_cost);
        } else if (s.equals("V")) {
            return -move_cost;
        } else {
            return 0;
        }

    }

    public static int getLength() {
        return length;
    }
    public static int getWidth() {
        return width;
    }

    public static void setMaze(String[][] maze2) {
        for (int i=0; i < maze.length; i++) {
            for (int j=0; j < maze[i].length; j++) {
                maze[i][j] = maze2[i][j];
            }
        }
    }

    public static void initializeVS(String[][] maze) {
        for (int i=0; i < maze.length; i++) {
            for (int j=0; j < maze[i].length; j++) {
                vs[i][j] = getReward(maze[i][j]);
                vs_prime[i][j] = getReward(maze[i][j]);
            }
        }
    }

    //Runs value iteration for a given state
    public static double valueIteration(int state_x, int state_y) {
        double val_k;
        double max_action = -INF;
        //Check east
        if (state_x + 1 < width) {
            if (!maze[state_y][state_x+1].equals("O")) {
                double forward_val = vs_prime[state_y][state_x+1];
                double stay_val = vs_prime[state_y][state_x];
                double back_val = 0.0;
                if (state_x - 1 > -1) {
                    back_val = vs_prime[state_y][state_x-1];
                }
                double action = p_forward*forward_val + p_stay*stay_val
                        + p_backward*back_val;
                if (action > max_action) {
                    max_action = action;
                }
            }
        }

        //Check South
        if (state_y + 1 < length) {
            if (!maze[state_y+1][state_x].equals("O")) {
                double forward_val = vs_prime[state_y+1][state_x];
                double stay_val = vs_prime[state_y][state_x];
                double back_val = 0.0;
                if (state_y - 1 > -1) {
                    back_val = vs_prime[state_y-1][state_x];
                }
                double action = p_forward*forward_val + p_stay*stay_val
                        + p_backward*back_val;
                if (action > max_action) {
                    max_action = action;
                }
            }
        }

        //Check West
        if (state_x - 1 > -1) {
            if (!maze[state_y][state_x-1].equals("O")) {
                double forward_val = vs_prime[state_y][state_x-1];
                double stay_val = vs_prime[state_y][state_x];
                double back_val = 0.0;
                if (state_x + 1 < width) {
                    back_val = vs_prime[state_y][state_x+1];
                }
                double action = p_forward*forward_val + p_stay*stay_val
                        + p_backward*back_val;
                if (action > max_action) {
                    max_action = action;
                }
            }
        }
        //Check North
        if (state_y - 1 > 0) {
            if (!maze[state_y-1][state_x].equals("O")) {
                double forward_val = vs_prime[state_y-1][state_x];
                double stay_val = vs_prime[state_y][state_x];
                double back_val = 0.0;
                if (state_y + 1 < length) {
                    back_val = vs_prime[state_y+1][state_x];
                }
                double action = p_forward*forward_val + p_stay*stay_val
                        + p_backward*back_val;
                if (action > max_action) {
                    max_action = action;

                }
            }
        }
        val_k = getReward(maze[state_y][state_x]) + max_action;

        return val_k;
    }

    //Run VI until vs and vs' converge to within epsilon
    public static void runVI() {
        int numIterations = 0;
        double epsilon = INF; //start value for epsilon, epsilon is total error
        //Initialize vs and vs_prime
        initializeVS(maze);
        for(int i=0; i < length; i++) {
            System.out.println(Arrays.toString(vs[i]));
        }
        while (epsilon > 10) {
            epsilon = 0;
            for (int i=length-1; i >= 0; i--) {
                for (int j=width-1; j >= 0; j--) {
                    if (!maze[i][j].equals("O")) {
                        vs[i][j] = valueIteration(j, i);
                    }
                }
            }
            //copy vs into vs_prime for next iteration
            for (int i=length-1; i >= 0; i--) {
                for (int j=width-1; j >= 0; j--) {
                    epsilon += Math.abs(vs[i][j] - vs_prime[i][j]);
                    vs_prime[i][j] = vs[i][j];
                }
            }
            numIterations++;
        }
        System.out.println("Value Iteration Complete");
        System.out.println("Total Iterations: " + numIterations);
        System.out.print("[");
        for(int i=0; i < length; i++) {
            System.out.print("[");
            for(int j=0; j < width-1; j++) {
                System.out.printf("%.2f, ", vs[i][j]);
            }
            System.out.printf("%.2f]\n", vs[i][width-1]);
        }


    }

    public static void main(String[] args) throws Exception {
        String[][] newMaze;
        // open maze file
        // one can change the maze file here (make sure they are in the same directory)
        File myFile = new File("src/maze2.txt");
        //Read in maze file
        Scanner fileScan = new Scanner(myFile);
        length = fileScan.nextInt();
        width = fileScan.nextInt();
        newMaze = new String[length][width];
        for (int i=0; i < length; i++) {
            for (int j=0; j < width; j++) {
                newMaze[i][j] = fileScan.next().trim();
            }

        }
        fileScan.close();
        maze = new String[length][width];
        for (int i=0; i < newMaze.length; i++) {
            for (int j=0; j < newMaze[i].length; j++) {
                maze[i][j] = newMaze[i][j];
            }
        }
        vs = new double[length][width];
        vs_prime = new double[length][width];
        es = new double[length][width];
        for (int i=0; i < length; i++) {
            for (int j=0; j < width; j++) {
                vs[i][j] = 0;
                vs_prime[i][j] = 0;
                es[i][j] = 0;
            }
        }
        valueIterationRewards = new ArrayList<Double>();
        //Perform operations here

        //Run Value Iteration
        runVI();


        //Plot Values
        /*
        final XYSeriesDemo demo = new XYSeriesDemo("Value Iteration Convergence");
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);
        */
    }
}
