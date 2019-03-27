/**
 * Title:
 * Description:
 * @author: Marvin Newlin
 * @version 1.0
 */

public class WorkBoard extends Board {
	static final int INF = 10000;
	Move best_move = null;  // Put your best move in here!
    int start_depth = 0;
    int totalNodesSearched = 0;
    int numLeafNodes = 0;
    boolean stoptime = true;
    //Changed this value to 5000L in order to allow search to progress past depth of 3
    public long searchtime = 5000L;

    //This int array comes from Mark Winand's Thesis chapter 3 p. 22
    //in his section on centralisation
    private int pieceSquareTable[][] = {
                                      {-80, -25, -20, -20, -20, -20, -25, -80},
                                      {-25, 10, 10, 10, 10, 10, 10, -25},
                                      {-20, 10, 25, 25, 25, 25, 10, -20},
                                      { -20, 10, 25, 50, 50, 25, 10, -20},
                                      {-20, 10, 25, 50, 50, 25, 10, -20},
                                      { -20, 10, 25, 25, 25, 25, 10, -20},
                                      { -25, 10, 10, 10, 10, 10, 10, -25},
                                      {-80, -25, -20, -20, -20, -20, -25, -80}
                                                                                };

    public WorkBoard() {
    }

    public WorkBoard(WorkBoard w) {
    	super(w);
    }

    /**
     * This is where your board evaluator will go. This function will be called
     * from min_max
     *
     * @param b the working copy of the board
     * @param player the player (white or black) whose status we are evaluating
     * @return int calculated heuristic value of the board
     */
    int h_value(WorkBoard b, int player) {

      int numWhitePieces = 0;
      int numBlackPieces = 0;
      double whiteAvg = 0;
      double blackAvg = 0;
      int whiteSum = 0;
      int blackSum = 0;

      for (int i=0; i < b.piece_list.length; i++) {
          if (b.piece_list[i].owner == PLAYER_WHITE) {
            for (Piece p = b.piece_list[i]; p != null; p = p.next) {
                numWhitePieces++;
                whiteSum += b.pieceSquareTable[p.y][p.x];
            }

          } else {
              for (Piece p = b.piece_list[i]; p != null; p = p.next) {
                  numBlackPieces++;
                  blackSum += b.pieceSquareTable[p.y][p.x];
              }
          }
      }

      whiteAvg = (double)whiteSum/numWhitePieces;
      blackAvg = (double)blackSum/numBlackPieces;
      double whiteHVal = Math.floor(blackAvg - whiteAvg);
      double blackHVal = Math.floor(whiteAvg - blackAvg);
      if (player == PLAYER_WHITE) {
          return (int) whiteHVal;
      } else {
          return (int) blackHVal;
      }

    }


    /**
     * This is where you will write min-max alpha-beta search. Note that the
     * Board class maintains a predecessor, so you don't have to deal with
     * keeping up with dynamic memory allocation.
     * The function takes the search depth, and returns the maximum value from
     * the search tree given the board and depth.
     *
     * @parama depth int the depth of the search to conduct
     * @return maximum heuristic board found value
     */
    Move min_max_AB(int depth, int alpha, int beta) {
      //Generate instance of current state
      WorkBoard currBoard = new WorkBoard(this);
      //Generate the current moves
      //Go through each move in the current board
      int maxPlayer = currBoard.to_move;
      Move moveList = currBoard.genMoves();
      moveList.value = -INF;
      Move best = moveList;
      best.value = -INF;
      Move tempMove = null;
      for (tempMove = moveList; tempMove != null; tempMove = tempMove.next) {
          currBoard.makeMove(tempMove);
          tempMove.value = MaxValue(currBoard, depth, alpha, beta, maxPlayer);
          if (tempMove.value >= best.value) {
              best = tempMove;
          }
          currBoard.reverseMove(tempMove);
      }
      return best;
    }

    /**
     *
     * @param b The board representing the current state
     * @param depth the depth to search to
     * @param alpha the max value for pruning
     * @param beta the min value for pruning
     * @param maxPlayer the player whose utility we are maximizing
     * @return int the maximum value
     */
    private int MaxValue(WorkBoard b, int depth, int alpha, int beta, int maxPlayer) {
        totalNodesSearched++;
        //Check Depth
        if (depth == 0) {
            return h_value(b, maxPlayer);
        }
        //Terminal check for max_player
        if (connected(maxPlayer)) {
            numLeafNodes++;
            return h_value(b, maxPlayer);
        }
        //Terminal Check for min player
        if (connected(opponent(maxPlayer))) {
            numLeafNodes++;
            return -h_value(b, maxPlayer);
        }
        Move moveList = b.genMoves();
        int v = -INF + depth;
        while(moveList != null) {
            b.makeMove(moveList);
            v = Math.max(v, MinValue(b, depth-1, alpha, beta, maxPlayer));
            b.reverseMove(moveList);
            if (v >= beta) {
                return v;
            }
            alpha = Math.max(alpha, v);
            moveList = moveList.next;
        }
        return v;
    }

    /**
     *
     * @param b The board representing the current state
     * @param depth the depth to search to
     * @param alpha the max value for pruning
     * @param beta the min value for pruning
     * @param maxPlayer the player whose utility we are maximizing
     * @return int the minimum value
     */
    private int MinValue(WorkBoard b,  int depth, int alpha, int beta, int maxPlayer) {
        totalNodesSearched++;
        //Check Depth
        if (depth == 0) {
            return h_value(b, opponent(maxPlayer));
        }
        //Terminal check for max_player
        if (connected(maxPlayer)) {
            return -h_value(b, opponent(maxPlayer));
        }
        //Terminal check for min player
        else if (connected(opponent(maxPlayer))) {
            return h_value(b, opponent(maxPlayer));
        }
        Move moveList = b.genMoves();
        int v = INF - depth;
        while (moveList != null) {
            b.makeMove(moveList);
            v = Math.min(v, MaxValue(b, depth-1, alpha, beta, maxPlayer));
            b.reverseMove(moveList);
            if (v <= alpha) {
                return v;
            }
            beta = Math.min(beta, v);
            moveList = moveList.next;
        }
        return v;
    }
       
    /**
     * This function is called to perform search. All it does is call min_max.
     *
     * @param depth int the depth to conduct search
     */
    void bestMove(int depth) {
      best_move = null;
      int runningNodeTotal = 0;
      totalNodesSearched = numLeafNodes = moveCount = 0;
      start_depth = 1;
      int i = 1;
      long startTime = System.currentTimeMillis();
      long elapsedTime = 0;
      long currentPeriod = 0;
      long previousPeriod = 0;
      stoptime = false;

      while ( i <= depth && !stoptime) {
        totalNodesSearched = numLeafNodes = moveCount = 0;
        start_depth = i;

        best_move = min_max_AB(i, -INF, INF); // Min-Max alpha beta

        elapsedTime = System.currentTimeMillis()-startTime;
        currentPeriod = elapsedTime-previousPeriod;
        double rate = 0;
        if ( i > 3 && previousPeriod > 50 )
          rate = (currentPeriod - previousPeriod)/previousPeriod;

        runningNodeTotal += totalNodesSearched;
        System.out.println("Depth: " + i +" Time: " + elapsedTime/1000.0 + " Nodes Searched: " + totalNodesSearched + " Leaf Nodes: " + numLeafNodes);

        // increment indexes: increase by two to avoid swapping between optimistic and pessimistic results
        i=i+2;
        
        if ( (elapsedTime+(rate+1.0)*currentPeriod) > searchtime )
          stoptime = true;
      }

      System.out.println("Nodes per Second = " + runningNodeTotal/(elapsedTime/1000.0));
      if (best_move == null  || best_move.piece == null) {
        throw new Error ("No Move Available - Search Error!");
      }
    }
}
