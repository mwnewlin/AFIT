
import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class Newlin_TD {
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
    //TD Parameters
    private static double  alpha = 0.2;
    private static double gamma = 0.8;
    private static double lambda = 0.5;
    private static double delta;

    static ArrayList<Double> TD0Rewards;
    static ArrayList<Double> TDLambdaRewards;

    public static double getReward(String s) {
        if (s.equals("G")) {
            return 100-move_cost;
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
                es[i][j] = 0;
            }
        }
    }

    public static Action getMaxAction(int state_x, int state_y) {
        double val_k;
        double max_action = -INF;
        int x = state_x;
        int y = state_y;
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
                    x = state_x + 1;
                    y = state_y;
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
                    x = state_x;
                    y = state_y + 1;
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
                    x = state_x - 1;
                    y = state_y;
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
                    x = state_x;
                    y = state_y - 1;
                }
            }
        }
        val_k = getReward(maze[state_y][state_x]) + max_action;
        Action max = new Action(x, y, val_k);
        return max;
    }



    /**
     * Runs TD0 from the provided start state
     * @param state_x the start x location
     * @param state_y the start y location
     * @return
     */
    public static double tdZero(int state_x, int state_y) {
        int numSteps = 0;
        boolean goal;
        while(numSteps < 50) {
            Action max = getMaxAction(state_x, state_y);
            vs[state_y][state_x] = vs[state_y][state_x] +
                    alpha * (getReward(maze[state_y][state_y]) +
                            (lambda * vs[max.getY()][max.getX()]) -
                            vs[state_y][state_x]);

            state_x = max.getX();
            state_y = max.getY();
            goal = maze[state_y][state_x].equals("G");
            if(goal) {
                break;
            }
            numSteps++;
        }
        return vs[state_y][state_x];
    }

    public static void trainTD0() {
        System.out.println("Training TD0");
        initializeVS(maze);
        Random r = new Random();
        int start_x = r.nextInt(length);
        int start_y = r.nextInt(length);
        while (maze[start_y][start_x].equals("O")) {
            start_x = r.nextInt(length);
            start_y = r.nextInt(length);
        }
        int numEpisodes = 200;
        for (int k=0; k < numEpisodes; k++) {
            double reward = tdZero(start_x, start_y);

        }
    }

    public static void evaluateTD0() {
        System.out.println("Evaluating TD0");
        //initializeVS(maze);
        int numEpisodes = 200;
        for (int k=0; k < numEpisodes; k++) {
            double reward = tdZero(width-1, length-1);
            TD0Rewards.add(reward);
        }
    }

    public static double tdLambda(int state_x, int state_y) {
        int numSteps = 0;
        boolean goal = maze[state_x][state_y].equals("G");
        while(numSteps < 50) {
            Action max = getMaxAction(state_x, state_y);
            delta = max.getReward() + Math.abs(gamma*vs_prime[max.getY()][max.getX()]
                    - vs_prime[state_y][state_x]);
            es[state_y][state_x] = es[state_y][state_x] + 1;
            for (int i=0; i < length; i++) {
                for (int j=0; j < width; j++) {
                    vs[i][j] = vs_prime[i][j] +
                            alpha*delta*es[state_y][state_x];
                }
            }
            state_x = max.getX();
            state_y = max.getY();
            goal = maze[state_y][state_x].equals("G");
            if(goal) {
                break;
            }
            numSteps++;
        }
        return vs[state_y][state_x];
    }

    public static void trainTDLambda() {
        System.out.println("Training TD Lambda");
        initializeVS(maze);
        Random r = new Random();
        int start_x = r.nextInt(length);
        int start_y = r.nextInt(length);
        while (maze[start_y][start_x].equals("O")) {
            start_x = r.nextInt(length);
            start_y = r.nextInt(length);
        }
        int numEpisodes = 200;
        for (int k=0; k < numEpisodes; k++) {
            double reward = tdLambda(start_x, start_y);
        }
    }

    public static void evaluateTDLambda() {
        System.out.println("Evaluating TD Lambda");
        //initializeVS(maze);
        int numEpisodes = 200;
        for (int k=0; k < numEpisodes; k++) {
            double reward = tdLambda(width-1, length-1);
            TDLambdaRewards.add(-reward);
        }
    }


    public static void main (String[] args ) throws Exception{
        String[][] newMaze;
        // open maze file
        // one can change the maze file here (make sure they are in the same directory)
        File myFile = new File("src/maze3.txt");
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
        TD0Rewards = new ArrayList<>();
        TDLambdaRewards = new ArrayList<>();

        //Run Operations
        trainTD0();
        evaluateTD0();
        trainTDLambda();
        evaluateTDLambda();

        //Plot Values
        System.out.println("Time to Plot");
        System.out.println(TD0Rewards.toString());
        System.out.println(TDLambdaRewards.toString());


        final XYSeriesDemo demo = new XYSeriesDemo("TD Lambda convergence", TDLambdaRewards);
        demo.pack();
        RefineryUtilities.centerFrameOnScreen(demo);
        demo.setVisible(true);

        final XYSeriesDemo demo2 = new XYSeriesDemo("TD Zero convergence", TD0Rewards);
        demo2.pack();
        RefineryUtilities.centerFrameOnScreen(demo2);
        demo2.setVisible(true);
    }


}


