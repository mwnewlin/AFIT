package rushhour.afit.edu;
/**
 * <p>Title: Rush Hour</p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2005</p>
 * <p>Company: AFIT</p>
 * @author not attributable
 * @version 1.0
 */

public class Piece {
    public static final int NORTH_SOUTH = 1;
    public static final int EAST_WEST = 2;
    public static final int NSEW = 3;

    public String name;        // Name of the piece
    public int move_direction; // direction piece can move
    public int x, y;           // location (lower left corner)
    public int dx, dy;         // size of piece

    /**
     * Default constructor
     */
    public Piece() {
      x = y = dx = dy = -1;
      move_direction = -1;
      name = "empty";
    }

    /**
     * Main constructor
     * 
     * @param x1
     * @param y1
     * @param dx1 - x size
     * @param dy1 - y size
     * @param md  - move_direction (see const list)
     * @param n   - name
     */
    public Piece( int x1, int y1, int dx1, int dy1, int md, String n ){
      x = x1;
      y = y1 + 1 - dy1; // Adjust to lower left corner
      dx = dx1;
      dy = dy1;
      move_direction = md;
      name = n;
    }

    /**
     * Copy constructor
     *
     * @param p Piece
     */
    public Piece(Piece p) {
    	x = p.x;
    	y = p.y;
    	dx = p.dx;
    	dy = p.dy;
    	move_direction = p.move_direction;
    	name = p.name;
    }
}
