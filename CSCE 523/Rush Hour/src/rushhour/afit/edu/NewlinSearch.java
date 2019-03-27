/**
 * @author Marvin Newlin
 * CSCE 523 Assignment 1
 * 16 Jan 19
 */
package rushhour.afit.edu;
import java.util.*;
public class NewlinSearch implements Search {

    private long nodes;
    private Board board;
    //Standard FIFO Queue for BFS
    private LinkedList<Board> open_list;
    private LinkedList<Board> visited;

    public NewlinSearch(Board b) {
        this.board = new Board(b);
        nodes = 0;

    }

    /**
     * The findMoves method will be where the search code actually goes. It is
     * expecting a return of the move list to the goal. The Board objects keep
     * track of the moves which got you to the goal so all you need to do is once
     * the goal is found return the move_list for the goal Board.
     *
     * @return
     */
    @Override
    public Move findMoves() {
        open_list = new LinkedList<>();
        visited = new LinkedList<Board>();
        Board currBoard = new Board(this.board);
        open_list.add(currBoard);

        boolean done = false;
        while (!done) {
            currBoard = open_list.removeFirst();
            boolean atGoal = goalCheck(currBoard);
            if ( atGoal || currBoard == null) {
                done = true;
                return currBoard.move_list;
            }
            visited.add(currBoard);
            this.nodes = visited.size();
            Move possMoves = currBoard.genMoves();

            //Iterate through possible moves, make the move and check
            //if the move puts us into a previously visited state
            //If it does, ignore it and move on
            while (possMoves != null) {
                Board temp = new Board(currBoard);
                temp.makeMove(possMoves);
                //Check to see if the move will put us into a board we've already visited
                //or if it puts us into a board in the open list
                //Add board to queue for exploration if it isn't in the visited list or open list
                if (!visited.contains(temp) && !open_list.contains(temp)) {
                    open_list.add(temp);
                }
                //Update move list
                possMoves = possMoves.next;
            }
        }
        return null;
    }

    /**
     * Inside your search you should also keep an incrementor around which keeps
     * track of the number of nodes expanded. This method just returns that count.
     *
     * @return
     */
    @Override
    public long nodeCount() {
        return this.nodes;
    }

    /**
     * Checks for Goal Condition: Red Piece at board exit
     * @param b
     * @return
     */
    private boolean goalCheck(Board b) {
        return b.goalCheck(b);
    }
}
