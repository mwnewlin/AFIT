package rushhour.afit.edu;
/**
 * <p>Title: Rush Hour</p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2005</p>
 * <p>Company: AFIT</p>
 * @author not attributable
 * @version 1.0
 */
import java.io.*;

public class RushHour
{
    public static String newBoard[][];//array for boards

    public static void main(String[] args) throws IOException {
      Board board;//workspace
      Search search; // the search algorithm
      long startTime; //timer
      char x[] = new char[2]; //temp variable used for file parsing

      //open board data file
      File inputFile = new File("killers.txt");
      FileReader in = new FileReader(inputFile);

      //get number of boards from first line
      int numBoards = in.read() - 48;
      //used for output
      int boardCounter = 0;
      //read carrige return
      in.read();

      //main prog loop
      while (numBoards > 0) {
        //increment board counter
        boardCounter++;

        //generate new space to store board
        newBoard = new String[Board.BOARD_SIZE][Board.BOARD_SIZE];

        //nested loop to parse file into an array
        for (int i = 0; i < Board.BOARD_SIZE; i++) {
          in.read(); //new line characters
          for (int j = 0; j < Board.BOARD_SIZE; j++) {
            in.read(x,0,2);
            newBoard[i][j] = new String(x);
          }
          in.read(); //carrige return
        }

        //decrement boards counter
        numBoards = numBoards - 1;

        //delete space between board if needed
        if (numBoards > 0) {
          //space between boards
          in.read(); //start
          in.read(); //return
        }

        //create new workspace
        board = new Board();
        // fill the board space
        board.importBoard(newBoard);

        // HERE: on the following line you need to have a search created, sending it the 
        // initial node.
        search = new NewlinSearch(board);

        
        //start timer
        startTime = System.currentTimeMillis();
        System.out.println("Started board " + boardCounter);

        //find moves to get out of jungle
        Move result = search.findMoves();

        System.out.println( "Board: " + boardCounter + "  " + ( (float)(System.currentTimeMillis() - startTime) / 1000.0) +
                           " seconds" );
        System.out.println("Nodes Visited: " + search.nodeCount());

        // Before printing need to reverse the move list
        if ( result == null )
          System.out.println("No path found!");
        else {
          result = reverseMoves(result);
          boolean first = true;
          for (Move theMove = result; theMove != null; theMove = theMove.next ) {
            if ( first ) {
              System.out.print("Move " + board.piece_list[theMove.piece_index].name);
              first = false;
            }
            else
              System.out.print( "     " + board.piece_list[theMove.piece_index].name );
            if ( theMove.direction == Move.NORTH )
              System.out.println( " North " + theMove.spaces + " spaces");
            if ( theMove.direction == Move.SOUTH )
              System.out.println( " South " + theMove.spaces + " spaces");
            if ( theMove.direction == Move.EAST )
              System.out.println( " East " + theMove.spaces + " spaces");
            if ( theMove.direction == Move.WEST )
              System.out.println( " West " + theMove.spaces + " spaces");
          }
        }
        System.out.println();
      } //end loop

      //close board file
      in.close();
    }
    
    /**
     * reverseMoves method: reverses the ordering of a move list.
     * This is because during the search the moves are stored in reverse order
     * to make the addition of items less of a hassle.
     * 
     * @param b_m
     * @return
     */
    private static Move reverseMoves(Move b_m) {
      Move b_ptr, temp;
  	  
	  Move forward = new Move(b_m);
	  b_ptr = b_m;
	  while ( b_ptr.next != null ){
	    b_ptr = b_ptr.next;
	    temp = new Move(b_ptr);
	    temp.next = forward;
	    forward = temp;
      }
      return forward;
    }
}

