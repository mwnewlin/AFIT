import java.util.List;
import java.util.Vector;
import java.io.*;

/**
 * Title:
 * Description:
 * @author: Bert Peterson
 * @version 1.0
 */
public class Board {
    /**
     * Game state constants
     * @param GAME_OVER.
     * @param CONTINUE.
     * @param ILLEGAL_MOVE.
     * @param PLAYER_WHITE.
     * @param PLAYER_BLACK.
     */
    public static final int GAME_OVER = 1;
    public static final int CONTINUE = 0;
    public static final int ILLEGAL_MOVE = -1;
    public static final int MAKE_MOVE_FAILED = -2;
    static final int EMPTY_SQUARE = -1;
    static final int PLAYER_WHITE = 0;
    static final int PLAYER_BLACK = 1;
    static final int WHITE_CHECKER = PLAYER_WHITE;
    static final int BLACK_CHECKER = PLAYER_BLACK;
    static final int OBSERVER = -1;
    static final int NONE = 9;
    static final int BOARD_SIZE = 8;               // CHANGE THIS FOR SMALLER BOARDS AND EASIER DEBUGGING
    static final int BOARD_INDEX = BOARD_SIZE - 1; // this is used to cap loops, etc.
    static final int MAX_DEPTH = 250;

    public int game_state = CONTINUE;
    public int to_move = PLAYER_BLACK;
    public int serial = 1;

    // rules-specific game state
    int square[][] = new int[BOARD_SIZE][BOARD_SIZE];
    int saved_game_state[] = new int[MAX_DEPTH];
    int saved_serial[] = new int[MAX_DEPTH];
    // These three arrays contain the piece count on the vertical, horizontal,
    // and both diagonals - hopefully to speed move generation as per Winands.
    int vertical_count[] = new int[BOARD_SIZE];
    int horizontal_count[] = new int[BOARD_SIZE];
    int forward_diag_count[] = new int [BOARD_SIZE+BOARD_INDEX];
    int back_diag_count[] = new int [BOARD_SIZE+BOARD_INDEX];

    int quad[][][] = new int [2][BOARD_SIZE+1][BOARD_SIZE+1];
    int quadcount[][] = new int [2][6];

    // Maintaining the piece lists in addition to the boards really speeds up the
    // heuristic calculation.
    Piece piece_list[] = new Piece[2];

    // Data structures for move ordering.
    // killer_moves is the best move at each depth found so far
    // history_moves is the best move for every search performed rated by depth of search
    Move killer_moves[] = new Move[MAX_DEPTH];
    int history_moves[][][][][] = new int[2][BOARD_SIZE][BOARD_SIZE][BOARD_SIZE][BOARD_SIZE];

    int moveCount = 0;

    // Everybody's favorite manual debug variable.
    final static private boolean debug_move_ok = false;
    private static final boolean debug_try_move = false;
    private static final boolean debug_quads = false;

    /**
     * Board Constructor:
     * Generates the board, with all of the pieces in the start positions.
     * Two rows of black one at the top, one at the bottom, and two rows of white
     * one on the right, the other the left. No pieces in the corners.
     */
    public Board() {
    	initialize();
    }

    /**
     * Board Copy Constructor
     * Generates a new board copying the Board b information.
     *
     * @param   b Board object
     */
    public Board(Board b) {
        for (int i = 0; i < BOARD_SIZE; i++) {
          for (int j = 0; j < BOARD_SIZE; j++)
            square[i][j] = b.square[i][j];
          vertical_count[i] = b.vertical_count[i];
          horizontal_count[i] = b.horizontal_count[i];
          forward_diag_count[i] = b.forward_diag_count[i];
          back_diag_count[i] = b.back_diag_count[i];
          forward_diag_count[i+BOARD_INDEX] = b.forward_diag_count[i+BOARD_INDEX];
          back_diag_count[i+BOARD_INDEX] = b.back_diag_count[i+BOARD_INDEX];
        }
        to_move = b.to_move;
        game_state = b.game_state;
        serial = b.serial;
        refreshDataStructures(); // make the quad and piece lists match the board state.
    }

    /**
     * Initializes the board, setting all pieces in the starting configuration.
     * This is called on Board creation as well as whenever the game restarts to
     * reset the entire board state.
     *
     */
    public void initialize() {
    	to_move = PLAYER_BLACK;
        game_state = CONTINUE;
        serial = 1;
    	piece_list[BLACK_CHECKER] = null;
    	piece_list[WHITE_CHECKER] = null;
        for (int i = 0; i < BOARD_SIZE; i++)
          for (int j = 0; j < BOARD_SIZE; j++)
            square[i][j] = EMPTY_SQUARE;
        for (int i = 1; i < BOARD_INDEX; i++) {
          square[i][0] = BLACK_CHECKER;
          square[i][BOARD_INDEX] = BLACK_CHECKER;
          piece_list[BLACK_CHECKER] = new Piece(i,0, BLACK_CHECKER, piece_list[BLACK_CHECKER], null);
          piece_list[BLACK_CHECKER] = new Piece(i,BOARD_INDEX, BLACK_CHECKER, piece_list[BLACK_CHECKER], null);
          square[0][i] = WHITE_CHECKER;
          square[BOARD_INDEX][i] = WHITE_CHECKER;
          piece_list[WHITE_CHECKER] = new Piece(0,i, WHITE_CHECKER, piece_list[WHITE_CHECKER], null);
          piece_list[WHITE_CHECKER] = new Piece(BOARD_INDEX,i, WHITE_CHECKER, piece_list[WHITE_CHECKER], null);
          vertical_count[i] = 2;
          horizontal_count[i] = 2;
          forward_diag_count[i] = 2;
          forward_diag_count[i+BOARD_INDEX] = 2;
          back_diag_count[i] = 2;
          back_diag_count[i+BOARD_INDEX] = 2;
        }
        recountQuads();
        vertical_count[0] = BOARD_SIZE-2;
        vertical_count[BOARD_INDEX] = BOARD_SIZE-2;
        horizontal_count[0] = BOARD_SIZE-2;
        horizontal_count[BOARD_INDEX] = BOARD_SIZE-2;
        forward_diag_count[0] = 0;
        forward_diag_count[BOARD_INDEX] = 0;
        forward_diag_count[BOARD_INDEX*2] = 0;
        back_diag_count[0] = 0;
        back_diag_count[BOARD_INDEX] = 0;
        back_diag_count[BOARD_INDEX*2] = 0;
        
    }
    
    /**
     * Copy position strictly copies the piece positions from one board to the
     * other.
     *
     * @param b Board the board to be copied from.
     */
    public void copy_positions( Board b ) {
        for( int i = 0; i < BOARD_SIZE; i++)
            for ( int j = 0; j < BOARD_SIZE; j++ )
                square[i][j] = b.square[i][j];
    }

    /**
     * addQuad adds a piece to the quad of player side in square x, y
     *
     * @param x int
     * @param y int
     * @param side int
     * @param number int
     * @return int
     */
    private int addQuad(int x, int y, int side, int number) {
        if (++number == 6)
            return 3;
        if (number == 2 && x >= 0 && x < BOARD_INDEX && y >= 0 && y < BOARD_INDEX
            && (square[x][y] == side && square[x + 1][y + 1] == side
                || square[x + 1][y] == side && square[x][y + 1] == side))
            return 5;
        return number;
    }

    /**
     * subtractQuad removes a piece from the quad of player side at x,y
     *
     * @param x int
     * @param y int
     * @param side int
     * @param number int
     * @return int
     */
    private int subtractQuad(int x, int y, int side, int number) {
        if (--number == 4)
            return 1;
        if (number == 2 && x >= 0 && x < BOARD_INDEX && y >= 0 && y < BOARD_INDEX
            && (square[x][y] == side && square[x + 1][y + 1] == side
                || square[x + 1][y] == side && square[x][y + 1] == side))
            return 5;
        return number;
    }

    /**
     * quad_value counts the pieces in the board that would be covered by a
     * quad at x,y of player using 5 to represent a diagonal.
     *
     * @param x int
     * @param y int
     * @param player int
     * @return int
     */
    private int quad_value(int x, int y, int player) {
      int counter = 0;
      if (checker_of(player,x,y))
        counter++;
      if (checker_of(player,x,y+1))
        counter++;
      if (checker_of(player,x+1,y))
        counter++;
      if (checker_of(player,x+1,y+1))
        counter++;
      if (counter == 2 && ((checker_of(player,x,y) && checker_of(player,x+1,y+1))
                           || (checker_of(player,x+1,y) && checker_of(player,x,y+1))))
        return 5;
      return counter;
    }

    /**
     * Outputs the opposite color of the player color sent in.
     *
     * @param player Integer Player's color.
     * @return Integer [PLAYER_WHITE,PLAYER_BLACK]
     */
    static final int opponent(int player) {
        if (player == PLAYER_WHITE)
            return PLAYER_BLACK;
        if (player == PLAYER_BLACK)
            return PLAYER_WHITE;
        throw new Error("internal error: bad player " + player);
    }

    /**
     * Outputs whether the checker in location [x,y] is owned by the player.
     *
     * @param player int The player to test [PLAYER_WHITE,PLAYER_BLACK]
     * @param x int The column value to test.
     * @param y int The row value to test.
     * @return final boolean True if player present
     */
    public final boolean checker_of( int player, int x, int y ) {
        if ( x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE )
          return false;
        if ( square[x][y] != player )
          return false;
        return true;
    }

    // see declaration of BLACK_CHECKER, WHITE_CHECKER
    static final int checker_of(int player) {
        return player;
    }
    static final int owner_of(int checker) {
        return checker;
    }

    /**
     * Compares the Board b object with this Board.
     *
     * @param b Board The board to be compared with this one.
     * @return boolean True if boards are the same, else false.
     */
    boolean same_position(Board b) {
        if (to_move != b.to_move)
            return false;
        for (int i = 0; i < BOARD_SIZE; i++)
            for (int j = 0; j < BOARD_SIZE; j++)
                if (square[i][j] != b.square[i][j])
                    return false;
        return true;
    }

    /**
     * Test to determine if nothing has changed on the Board, for a draw.
     *
     * @return boolean True if the Board didn't change on the last move.
     */
/*    boolean repeated_position() {
        Board p = predecessor;
        while(p != null) {
            if (p.same_position(this))
                return true;
            p = p.predecessor;
        }
        return false;
    }
*/
    /**
     * Returns the sign of the number passed in.
     *
     * @param x int value to reverse.
     * @return int 1 or -1 for positive or negative.
     */
    static final int sgn(int x) {
        if (x > 0)
            return 1;
        if (x < 0)
            return -1;
        return 0;
    }

    /**
     * Verify that the value passed in is inside the range of the board [0..BOARD_INDEX]
     *
     * @param x int Value to test.
     * @return boolean true If INVALID false if valid.
     */
    static final boolean clipped(int x) {
        if (x >= BOARD_SIZE)
            return true;
        if (x < 0)
            return true;
        return false;
    }

    /**
     * Calculates the distance a piece can move, by traversing the line through
     * the location [x,y] along line [dx,dy] and counting the pieces on the line.
     *
     * @param x int start column location
     * @param y int start row locations
     * @param dx int the rate of change in column location.
     * @param dy int the rate of change in row location.
     * @return int the distance that can be traveled along this line.
     */
    private int dist(int x, int y, int dx, int dy) {
      if (dx == 0 )
        return vertical_count[x];
      if ( dy == 0 )
        return horizontal_count[y];
      if ( dx+dy == 0 )
        return back_diag_count[x+y];
      return forward_diag_count[x+(BOARD_INDEX-y)];
    }

    /**
     * Test to determine if a move along a line dx,dy of distance d would be
     * blocked by an opponents piece
     *
     * @param m Move the move to validate.
     * @param dx int the rate of change in column location.
     * @param dy int the rate of change in row location.
     * @param d int the distance being traveled.
     * @return boolean true if the move is blocked by an opponent piece.
     */
    private boolean blocked(Move m, int dx, int dy, int d) {
    	try {
          for (int q = 1; q < d; q++) {
            int xx = m.x1 + q * dx;
            int yy = m.y1 + q * dy;
            if (square[xx][yy] == opponent(square[m.x1][m.y1]))
                return true;
          }
        } catch(Error e){}
        return false;
    }

    /**
     * Method checks to determine if the move m is valid for this board.
     * By checking if the move goes off the board, then if it lands on a checker
     * of it's own, then checks if the move is blocked by opponent piece, else OK.
     *
     * @param m Move
     * @return Boolean [true, false] = [valid, invalid].
     */
    protected boolean move_ok(Move m) {
        int dx = sgn(m.x2 - m.x1);
        int dy = sgn(m.y2 - m.y1);
        if (debug_move_ok)
            System.out.println("entering move_ok(): dir " +
                               dx + ", " + dy);
        if (clipped(m.x2) || clipped(m.y2)) {
            if (debug_move_ok)
                System.out.println("leaving move_ok(): clipped");
            return false;
        }
        int d = dist(m.x1, m.y1, dx, dy);
        if ((m.x2 - m.x1) != d * dx) {
            if (debug_move_ok)
                System.out.println("leaving move_ok(): bad x disp");
            return false;
        }
        if ((m.y2 - m.y1) != d * dy) {
            if (debug_move_ok)
                System.out.println("leaving move_ok(): bad y disp");
            return false;
        }
        if (blocked(m, dx, dy, d)) {
            if (debug_move_ok)
                System.out.println("leaving move_ok(): blocked");
            return false;
        }
        if (square[m.x2][m.y2] == square[m.x1][m.y1]) {
            if (debug_move_ok)
                System.out.println("leaving move_ok(): self-capture");
            return false;
        }
        if (square[m.x1][m.y1] != to_move || square[m.x1][m.y1] == opponent(to_move) ) {
            if (debug_move_ok) {
                System.out.println("leaving move_ok(): not player's piece " +to_move+" "+square[m.x1][m.y1]+" ["+m.x1+", "+m.y1);
                            throw new Error("Not my piece");}
            return false;
        }
        if (debug_move_ok)
            System.out.println("leaving move_ok(): success");
        return true;
    }

    /**
     * Creates a list of all of the possible valid moves and stores them in a
     * List. Used by search to get a list of all possible moves.
     *
     * @return List list of all valid moves.
     */
    public Move genMoves() {
      Move result = null;
      for ( Piece p = piece_list[to_move]; p != null; p = p.next ) {
        int i = p.x;
        int j = p.y;
        if ( i < 0 || j < 0 || square[p.x][p.y] != to_move ) // Sometimes the piece list gets corrupted trying to stem the tide because I can't track the problem.
          continue;
        int pieces = vertical_count[i];
        if (j - pieces >= 0 && square[i][j - pieces] != to_move)
          result = blocked(result, i, j, 0, -1, pieces, p);
        if (j + pieces < BOARD_SIZE && square[i][j + pieces] != to_move)
          result = blocked(result, i, j, 0, 1, pieces, p);
        pieces = horizontal_count[j];
        if (i - pieces >= 0 && square[i - pieces][j] != to_move)
          result = blocked(result, i, j, -1, 0, pieces, p);
        if (i + pieces < BOARD_SIZE && square[i + pieces][j] != to_move)
          result = blocked(result, i, j, 1, 0, pieces, p);
        pieces = forward_diag_count[i +(BOARD_INDEX-j)];
        if (i + pieces < BOARD_SIZE && j + pieces < BOARD_SIZE && square[i + pieces][j + pieces] != to_move)
          result = blocked(result, i, j, 1, 1, pieces, p);
        if (i - pieces >= 0 && j - pieces >= 0 && square[i-pieces][j-pieces] != to_move)
          result = blocked(result, i, j, -1, -1, pieces, p);
        pieces = back_diag_count[i + j];
        if (i + pieces < BOARD_SIZE && j - pieces >= 0 && square[i + pieces][j - pieces] != to_move)
          result = blocked(result, i, j, 1, -1, pieces, p);
        if (i-pieces >= 0 && j+pieces < BOARD_SIZE && square[i - pieces][j + pieces] != to_move)
          result = blocked(result, i, j, -1, 1, pieces, p);
      }
      return result;
    }

    /**
     *  Performs the blocking test and generates the move data.
     *
     * @param result Move
     * @param row int
     * @param col int
     * @param x int
     * @param y int
     * @param pieces int
     * @param p Piece
     * @return Move
     */
    private Move blocked(Move result, int row, int col, int x, int y, int pieces , Piece p) {
        for (int i = 1; i < pieces; i++) {
            if (square[row + i * x][col + i * y] == opponent(to_move))
                return result;
        }
        Move m = new Move(row, col, row + x * pieces, col + y * pieces, 0 ,p );
        m.next = result;
        result = m;
        moveCount++;
        return result;
    }

    /**
     * Finds the piece in the piece_list of player side in position x, y
     *
     * @param x int
     * @param y int
     * @param side int
     * @return Piece
     */
    private Piece findPiece( int x, int y, int side ) {
      for( Piece p = piece_list[side]; p != null; p = p.next ) {
        if ( p.x == x && p.y == y )
          return p;
      }
      return null;
    }

    /**
     * Creates a list of all possible valid moves from location [i,j] and stores
     * them in a vector. Used by the GUI when a player selects a piece.
     *
     * @param i int x location value
     * @param j int y location value
     * @return Vector list of all valid moves from [i,j]
     */
    public Vector<Move> genMoves2( int i, int j) {
      List<Move> result = new Vector<Move>();
      Piece p = findPiece( i, j, to_move);
      if (square[i][j] == checker_of( to_move )){
        for ( int dx = -1; dx <= 1; dx++ )
          for (int dy = -1; dy <= 1; dy++ ) {
            if ( dx == 0 && dy == 0 )
              continue;
            int d = dist( i, j, dx, dy );
            Move m = new Move(i,j, i+d*dx,j+d*dy,0,p);
            if (move_ok(m))
              result.add(m);
          }
      }
      return (Vector<Move>)result;
    }

    /**
     * Quicksort sorting by move value high to low, decreasing
     *
     * @param list Move
     * @return Move
     */
    protected Move QuickSort(Move list) {
      if (list == null || list.next == null)
        return list;
      Move smaller = null;
      Move p = list;
      list = list.next;
      p.next = null;
      Move bigger = p;
      while (list != null) {
        if (p.value < list.value) {
          Move temp = list;
          list = list.next;
          temp.next = bigger;
          bigger = temp;
        } else {
          Move temp = list;
          list = list.next;
          temp.next = smaller;
          smaller = temp;
        }
      }
      smaller = QuickSort(smaller);
      bigger = QuickSort(bigger);
      p.next = smaller;
      return bigger;
    }

    /**
     * Provides values for each move the sort, search killer moves first, then based on
     * history moves and make moves to the outside edges last.
     *
     * @param list Move
     * @param side int
     * @param depth int
     * @return Move
     */
    protected Move moveOrdering(Move list, int side, int depth) {
      for (Move m = list; m != null; m = ( m).next) {
    	  // Assign move ordering values here
      }
      return QuickSort(list);
    }

    /**
     * Tests Board location [x,y] for the presence of a piece owned by the current
     * player. Used by the GUI for piece selection,
     *
     * @param x int column position.
     * @param y int row position.
     * @return boolean true if square occupied by current player
     */
    public boolean piece(int x, int y ) {
      try {
        if (square[x][y] == checker_of(to_move))
          return true;
        return false;
      }
      catch (Exception e) {
        return false;
      }
    }

    /**
     * Helper function to determine if the player won. Recursively calls itself
     * visiting each piece that is within one square of the piece passed in.
     * During visiting, a map is made that notes that it is reached, and the
     * the number of pieces reached via the piece is returned.
     *
     * @param side int the player [PLAYER_WHITE, PLAYER_BLACK]
     * @param x int column location of a player's piece
     * @param y int row location of a player's piece
     * @param map boolean[][] a map of the board of the reachable pieces
     * @return int the total number of pieces reachable from this piece.
     */
    int map_component(int side, int x, int y, boolean map[][]) {
        int total = 1;
        map[x][y] = true;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0)
                    continue;
                int nx = x + dx;
                int ny = y + dy;
                if (clipped(nx) || clipped(ny))
                    continue;
                if (square[nx][ny] != checker_of(side))
                    continue;
                if (map[nx][ny])
                    continue;
                total += map_component(side, nx, ny, map);
            }
        return total;
    }

    /**
     * Are all of this sides pieces connected, did the player win? First, find
     * a piece of the players color then call map_component, which counts the
     * number of pieces connected to this one. Then test to make sure that the
     * map made from visiting all pieces connected to the first one found are
     * all the pieces the player has. If all of the pieces are accounted for
     * the player wins.
     *
     * @param side int the player [PLAYER_WHITE, PLAYER_BLACK]
     * @return boolean true if all pieces connected, false if not.
     */
    boolean connected(int side) {
      int euler;

      // Before doing the full check just check the Euler number for this side.
      euler = (quadcount[side][1]-quadcount[side][3]-2*quadcount[side][5])/4;
      if ( euler > 1 )
        return false;

      Piece plist = null;
      int count = 0;
      for (Piece p = piece_list[side]; p != null; p = p.next) {
        plist =  new Piece(p.x, p.y, side, plist, null);
        count++;
      }
      Piece Connectlist = plist;
      plist = plist.next;
      Connectlist.next = null;
      boolean connection = true;
      while (count > 1 && plist != null && connection) {
          connection = false;
          Piece p = plist;
          for (Piece plist2 = Connectlist; plist2 != null && !connection;
               plist2 = plist2.next) {
              if ((p.x == plist2.x + 1 || p.x == plist2.x - 1
                   || p.x == plist2.x)
                  && (p.y == plist2.y || p.y == plist2.y + 1
                      || p.y == plist2.y - 1)) {
                  plist = plist.next;
                  p.next = Connectlist;
                  Connectlist = p;
                  connection = true;
              }
          }
          if (!connection) {
              Piece tracker = plist;
              for (p = plist.next; p != null && !connection; p = p.next) {
                  for (Piece plist2 = Connectlist; plist2 != null && !connection;
                       plist2 = plist2.next) {
                      if ((p.x == plist2.x + 1 || p.x == plist2.x - 1
                           || p.x == plist2.x)
                          && (p.y == plist2.y || p.y == plist2.y + 1
                              || p.y == plist2.y - 1)) {
                          tracker.next = p.next;
                          p.next = Connectlist;
                          Connectlist = p;
                          connection = true;
                      }
                  }
                  if (!connection)
                      tracker = p;
              }
          }
      }
      return connection;
    }

    /**
     * Execute the actual moving of a piece. In this game, they are easy.
     *
     * @param m Move the move to execute.
     */
    protected void makeMove(Move m) {
      // Check to see if this was a capture
      int x1 = m.x1;
      int y1 = m.y1;
      int x2 = m.x2;
      int y2 = m.y2;
      m.captured = square[x2][y2];
      int side = square[x1][y1];
      // reduce quad counts for the from and to locations then update the quads
      // and recalculate the quad values.
      subtractQuad_numbers(x1, y1, side);
      subtractQuad_numbers(x2, y2, side);

      // If we are only moving one square up, down or left, right need to ensure
      // that the quadcounts do not get off.
      Piece quad_list = null;
      if (Math.abs(x1 - x2) == 1 && (y1 - y2) == 0) {
        int max_row = Math.max(x1, x2);
        quad_list = new Piece(max_row, y1, side);
        quad_list.next = new Piece(max_row, y1 + 1, side);
      } else if ((x1 - x2) == 0 && Math.abs(y1 - y2) == 1) {
        int max_col = Math.max(y1, y2);
        quad_list = new Piece(x1, max_col, side);
        quad_list.next = new Piece(x1 + 1, max_col, side);
      }
      for (Piece p = quad_list; p != null; p = p.next)
        quadcount[side][quad[side][p.x][p.y]]++;

     // SIMPLE
      m.piece = findPiece( x1, y1, side);
      m.piece.x = m.x2;
      m.piece.y = m.y2;
      square[x2][y2] = square[x1][y1];
      square[x1][y1] = EMPTY_SQUARE;
      subtractLines(x1, y1);

      // QUAD
      
      // Now update the quad information
      removeQuad(x1, y1, side);
      addQuad(x2, y2, side);
      // If this was a capture do the same for the opponent
      if (m.captured != EMPTY_SQUARE) {
        piece_list[m.captured] = deletePiece(piece_list[m.captured], x2, y2, m);
        subtractQuad_numbers(x2, y2, m.captured);
        removeQuad(x2, y2, m.captured);
        addQuad_numbers(x2, y2, m.captured);
      } else
        addLines(x2, y2);
      // Here is the recalculating of the quad counts
      addQuad_numbers(x2, y2, side);
      addQuad_numbers(x1, y1, side);
      for (Piece p = quad_list; p != null; p = p.next)
        quadcount[side][quad[side][p.x][p.y]]--;
      to_move = opponent(to_move);
      if (debug_quads)
        quadCheck();
    }

    /**
     * Execute the removal of an action
     *
     * @param m Move the move to remove, should be the last move executed.
     */
    public void reverseMove( Move m ) {
      int x1 = m.x1;
      int y1 = m.y1;
      int x2 = m.x2;
      int y2 = m.y2;
      to_move = opponent(to_move);
      m.piece.x = x1;
      m.piece.y = y1;
      square[x1][y1] = square[x2][y2];
      square[x2][y2] = m.captured;
      int side = square[x1][y1];
      addLines(m.x1, m.y1);
      // If we are only moving one square up, down or left, right need to ensure
      // that the quadcounts do not get off.
      Piece quad_list = null;
      if (Math.abs(x1 - x2) == 1 && (y1 - y2) == 0) {
        int max_row = Math.max(x1, x2);
        quad_list = new Piece(max_row, y1, side);
        quad_list.next = new Piece(max_row, y1 + 1, side);
      } else if ((x1 - x2) == 0 && Math.abs(y1 - y2) == 1) {
        int max_col = Math.max(y1, y2);
        quad_list = new Piece(x1, max_col, side);
        quad_list.next = new Piece(x1 + 1, max_col, side);
      }
      for (Piece p = quad_list; p != null; p = p.next)
        quadcount[side][quad[side][p.x][p.y]]++;

      // QUAD
      subtractQuad_numbers(x1, y1, side);
      subtractQuad_numbers(x2, y2, side);

      // Now update the quad information, will update through the methods
      // instead of expending memory
      removeQuad(x2, y2, side);
      addQuad(x1, y1, side);

      if (m.captured != EMPTY_SQUARE) {
        piece_list[m.captured] = addPiece(piece_list[m.captured], m.captured_piece);
        subtractQuad_numbers(x2, y2, m.captured);
        addQuad(x2, y2, m.captured);
        addQuad_numbers(x2, y2, m.captured);
      } else
        subtractLines(x2, y2);

      addQuad_numbers(x2, y2, side);
      addQuad_numbers(x1, y1, side);
      for (Piece p = quad_list; p != null; p = p.next)
        quadcount[side][quad[side][p.x][p.y]]--;
      if (game_state == GAME_OVER)
    	  game_state = CONTINUE;
      if (debug_quads)
        quadCheck();
    }

    /**
     * Test to see if Move m can be made now, is available, is legal, and then
     * make it. Afterwards, test the effects of the move, is it the end of game,
     * or a repeated position, if not update the move to be the other player.
     *
     * @param m Move the move to try.
     * @return int result of the move [ILLEGAL_MOVE, GAME_OVER, CONTINUE]
     */
    public int try_move(Move m) {
        if (debug_try_move)
            System.err.println("entering try_move()");
        if (game_state != CONTINUE) {
            if (debug_try_move)
                System.err.println("leaving try_move(): move after game over");
            return ILLEGAL_MOVE;
        }
        /* This checking of a connected prior to making the move was causing this
           method to return without making the move. Because of this the
           reverse move that is called on return caused the board state to be
           corrupted. */
        if (connected(PLAYER_BLACK) || connected(PLAYER_WHITE)) { // !has_moves()) {
            game_state = GAME_OVER;
            to_move = opponent(to_move);
            if (debug_try_move)
                System.err.println("leaving try_move(): no legal moves");
            return GAME_OVER;
        }
        // There is no move to make. Results in game over and loss for this player.
        if (m.x1 == 0 && m.x2 == 0 && m.y1 ==0 && m.y2 == 0) {
            game_state = GAME_OVER;
            to_move = opponent(to_move);
            if (debug_try_move)
                System.err.println("leaving try_move(): no legal moves");
            return GAME_OVER;
        }
        if (!move_ok(m)) {
            if (debug_try_move)
                System.err.println("leaving try_move(): illegal move");
            return ILLEGAL_MOVE;
        }
        if (debug_try_move)
            System.err.println("move ok");

        makeMove(m);

        if (connected(to_move)) {
            game_state = GAME_OVER;
            if (debug_try_move)
                System.err.println("leaving try_move(): move connected");
            return GAME_OVER;
        }
        if (connected(opponent(to_move))) {
            game_state = GAME_OVER;
            if (debug_try_move)
                System.err.println("leaving try_move(): move connected opponent");
            return GAME_OVER;
        }
        /*  Until this check is implemented this isn't required.
        if (repeated_position()) {
            game_state = GAME_OVER;
            if (debug_try_move)
                System.err.println("leaving try_move(): repeat draw");
            return GAME_OVER;
        } */
        if (to_move == PLAYER_BLACK)
            serial++;
        if (debug_try_move)
            System.err.println("leaving try_move(): continue game");
        return CONTINUE;
    }

    /**
     * Ask to see who won the game.
     *
     * @return int the winner [PLAYER_WHITE,PLAYER_BLACK,OBSERVER]
     */
    public int referee() {
        if (connected(PLAYER_BLACK))
          return PLAYER_BLACK;
        if (connected(PLAYER_WHITE))
          return PLAYER_WHITE;
        return OBSERVER;
    }

    /**
     * This print method is used for the demon to output the board state after
     * each move on debugging.
     *
     * @param s PrintStream
     */
    public void print(PrintStream s) {
	  s.print("Friendly game: ");
	  s.print(serial);
      s.print(" ");
	  if (game_state == GAME_OVER)
	    s.print("*");
	  else if (to_move == PLAYER_WHITE)
	    s.print("white");
	  else
	    s.print("black");
	  s.print("\r\n");
	  s.flush();
      // Print the board state.
	  for (int j = BOARD_INDEX; j >= 0; --j) {
	    for (int i = 0; i < BOARD_SIZE; i++)
		  switch (square[i][j]) {
		    case EMPTY_SQUARE:  s.print("."); break;
		    case BLACK_CHECKER:  s.print("b"); break;
		    case WHITE_CHECKER:  s.print("w"); break;
		    default: s.print("?");
		  }
	    s.print("\r\n");
	  }
	  s.flush();
    }

    /**
     * Reads the board layout from a string with 'b' for black, 'w' for white
     * and 'X' for a blank into the board data structures.
     *
     * @param b String
     */
    public void new_layout(String b ) {
      System.out.println(b);
      for ( int i = 0; i < BOARD_SIZE; i++ )
        for ( int j = 0; j < BOARD_SIZE; j++ ) {
          if ( b.charAt( i*BOARD_SIZE+j ) == 'b' )
            square[i][j] = BLACK_CHECKER;
          if ( b.charAt( i*BOARD_SIZE+j ) == 'w' )
            square[i][j] = WHITE_CHECKER;
          if ( b.charAt(i*BOARD_SIZE+j) == 'X' )
            square[i][j] = EMPTY_SQUARE;
        }
      refreshDataStructures();
    }

    public String vectorizeBoard() {
    
      String vector = new String();
      for ( int i = 0; i < BOARD_SIZE; i++ )
    	for ( int j = 0; j < BOARD_SIZE; j++ ) {
    	  vector += square[i][j];
    	}
      
      vector += to_move;
      return vector;
    }
    /**
     * Assuming the board representation in square is correct, recount the quads,
     * regenerate the piece lists, and update the line counts
     */
    protected void refreshDataStructures() {
      recountQuads();
      reloadPieceLists();
      for ( int i = 0; i < BOARD_SIZE; i++ ) {
        vertical_count[i] = horizontal_count[i] = forward_diag_count[i] = 0;
        back_diag_count[i] = forward_diag_count[i+BOARD_INDEX] = 0;
        back_diag_count[i+BOARD_INDEX] = 0;
      }
      for ( int i = 0; i < BOARD_SIZE; i++ )
        for ( int j = 0; j < BOARD_SIZE; j++ ){
          if (square[i][j] != EMPTY_SQUARE ) {
            vertical_count[i]++;
            horizontal_count[j]++;
            forward_diag_count[i + (BOARD_INDEX - j)]++;
            back_diag_count[i + j]++;
          }
        }
      System.out.println("DoneRefresh");
    }

    /**
     * Recounts and calibrates the quad values and quadcounts. Will be called by
     * reset layout and also will be used when loading a board from a file.
     */
    private void recountQuads() {
      for (int i = 0; i < 6; i++ )
        quadcount[0][i] = quadcount[1][i] = 0;
      for (int i = 0; i < BOARD_SIZE+1; i++) {
        for (int j = 0; j < BOARD_SIZE+1; j++) {
          quad[0][i][j] = quadValue(i, j, 0); // # of pieces in quad, 5 for diagonals per side
          quad[1][i][j] = quadValue(i, j, 1);
          quadcount[0][quad[0][i][j]]++; // quad counts per side
          quadcount[1][quad[1][i][j]]++;
        }
      }
    }

    /**
     * Helper function for recountQuads. For this quad, counts the number of
     * pieces in the quad of color side. If the count = 2 and they are a
     * diagonal return 5 otherwise return the count.
     *
     * @param x integer value for quad x
     * @param y integer value for quad y
     * @param side integer of player to calculate (PLAYER_WHITE, PLAYER_BLACK)
     * @return integer count of pieces of color side in quad or 5 if diagonal
     */
    private int quadValue(int x, int y, int side) {
        int counter = 0;
        if ( checker_of(side,x-1,y-1) )
            counter++;
        if ( checker_of(side,x,y-1) )
            counter++;
        if ( checker_of(side,x-1,y) )
            counter++;
        if ( checker_of(side,x,y) )
            counter++;
        if (counter == 2 && ((checker_of(side,x,y) && checker_of(side,x-1,y-1))
                             || (checker_of(side,x-1,y) && checker_of(side,x,y-1))))
            return 5;
        return counter;
    }

    /**
     * Uses the square to regenerate each player's piece list.
     */
    private void reloadPieceLists() {
      piece_list[0] = null;
      piece_list[1] = null;
      for ( int i = 0; i < BOARD_SIZE; i++ )
        for ( int j = 0; j < BOARD_SIZE; j++ ) {
          if ( square[i][j] == WHITE_CHECKER )
            piece_list[WHITE_CHECKER] = new Piece(i,j,WHITE_CHECKER,piece_list[WHITE_CHECKER], null);
          if ( square[i][j] == BLACK_CHECKER )
            piece_list[BLACK_CHECKER] = new Piece(i,j,BLACK_CHECKER,piece_list[BLACK_CHECKER], null);
        }
    }

    /**
     * Find and remove the piece at location x, y from the piece list 'list'
     *
     * @param list Piece
     * @param x int
     * @param y int
     * @param m Move
     * @return Piece
     */
    private Piece deletePiece(Piece list, int x, int y, Move m) {
        if (list.x == x && list.y == y) {
            m.captured_piece = list;
            list.next.prev = null;
            return list.next;
        }
        Piece old = list;
        for (Piece p = list.next; p != null; p = p.next) {
            if (p.x == x && p.y == y) {
                m.captured_piece = p;
                old.next = p.next;
                if (p.next != null)
                  p.next.prev = p.prev;
                return list;
            }
            old = p;
        }
        if ( m.captured_piece == null )
          System.out.println("Piece not found");
        return list;
    }

    /**
     * Insert piece p at the head of list 'list'
     *
     * @param list Piece
     * @param p Piece
     * @return Piece
     */
    private Piece addPiece(Piece list, Piece p) {
      if (p.prev == null)
         list = p;
      else
        p.prev.next = p;
      if (p.next != null)
        p.next.prev = p;
        return list;
    }

    /**
     * Add one to line counts for position x,y
     *
     * @param x int
     * @param y int
     */
    private void addLines(int x, int y) {
        horizontal_count[y]++;
        vertical_count[x]++;
        forward_diag_count[x + (BOARD_INDEX-y)]++;
        back_diag_count[x + y]++;
    }

    /**
     * Subtract one from the line counts for position x,y
     *
     * @param x int
     * @param y int
     */
    private void subtractLines(int x, int y) {
      horizontal_count[y]--;
      vertical_count[x]--;
      forward_diag_count[x + (BOARD_INDEX-y)]--;
      back_diag_count[x + y]--;
    }

    // Quad Updating
    private void addQuad(int x, int y, int side) {
      quad[side][x + 1][y + 1] = addQuad(x, y, side, quad[side][x + 1][y + 1]);
      quad[side][x + 1][y] = addQuad(x, y - 1, side, quad[side][x + 1][y]);
      quad[side][x][y + 1] = addQuad(x - 1, y, side, quad[side][x][y + 1]);
      quad[side][x][y] = addQuad(x - 1, y - 1, side, quad[side][x][y]);
    }

    private void removeQuad(int x, int y, int side) {
      quad[side][x + 1][y + 1] = subtractQuad(x, y, side, quad[side][x + 1][y + 1]);
      quad[side][x + 1][y] = subtractQuad(x, y - 1, side, quad[side][x + 1][y]);
      quad[side][x][y + 1] = subtractQuad(x - 1, y, side, quad[side][x][y + 1]);
      quad[side][x][y] = subtractQuad(x - 1, y - 1, side, quad[side][x][y]);
    }

    // Quad counts updating
    private void addQuad_numbers(int x, int y, int side) {
        quadcount[side][quad[side][x + 1][y + 1]]++;
        quadcount[side][quad[side][x + 1][y]]++;
        quadcount[side][quad[side][x][y + 1]]++;
        quadcount[side][quad[side][x][y]]++;
    }

    private void subtractQuad_numbers(int x, int y, int side) {
      quadcount[side][quad[side][x + 1][y + 1]]--;
      quadcount[side][quad[side][x + 1][y]]--;
      quadcount[side][quad[side][x][y + 1]]--;
      quadcount[side][quad[side][x][y]]--;
    }
    
	private void quadCheck() {
      int tempquad[][][] = new int [2][BOARD_SIZE+1][BOARD_SIZE+1];
      int tempquadcount[][] = new int [2][6];
      
      for (int i = -1; i < BOARD_SIZE; i++) {
        for (int j = -1; j < BOARD_SIZE; j++) {
          tempquad[PLAYER_BLACK][i + 1][j + 1] = quad_value(i, j, PLAYER_BLACK); // # of pieces in quad, 5 for diagonals per side
          tempquad[PLAYER_WHITE][i + 1][j + 1] = quad_value(i, j, PLAYER_WHITE);
          tempquadcount[PLAYER_BLACK][tempquad[PLAYER_BLACK][i + 1][j + 1]]++; // quad counts per side
          tempquadcount[PLAYER_WHITE][tempquad[PLAYER_WHITE][i + 1][j + 1]]++;
        }
      }
      for (int i=0; i <= BOARD_SIZE; i++) {
    	for (int j=0; j <= BOARD_SIZE; j++) {
    	  if (tempquad[PLAYER_BLACK][i][j] != quad[PLAYER_BLACK][i][j] || tempquad[PLAYER_WHITE][i][j] != quad[PLAYER_WHITE][i][j])
    		  System.out.println("QUADS DO NOT MATCH");
    	}
      }
      for (int i = 0; i < 6; i++){
    	  if (tempquadcount[PLAYER_BLACK][i] != quadcount[PLAYER_BLACK][i] || tempquadcount[PLAYER_WHITE][i] != quadcount[PLAYER_WHITE][i])
    		  System.out.println("QUADCOUNTS DO NOT MATCH");
      }
    }
}
