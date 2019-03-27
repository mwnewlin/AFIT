package rushhour.afit.edu;

import java.util.Arrays;
import java.util.Objects;

/**
 * <p>Title: Rush Hour</p>
 * <p>Description: </p>
 * <p>Copyright: Copyright (c) 2005</p>
 * <p>Company: AFIT</p>
 * @author not attributable
 * @version 1.0
 */

public class Board {
  static final int PIECE_COUNT = 30;
  static final int BOARD_SIZE = 6;
  static final int BOARD_INDEX = 5;  // 1 - BOARD_SIZE
  // Where the exit is so that this can be expanded to different board sizes
  static final int BOARD_EXIT_X = 6;
  static final int BOARD_EXIT_Y = 3;

  static final boolean SAFARI = false;

  public int theBoard[][];     //version of workspace array
  public Piece piece_list[];
  public int piece_count = 0;
  public Move move_list;

  /**
   * Board constructor
   */
  public Board() {
    theBoard = new int[BOARD_SIZE + 1][BOARD_SIZE];
    for (int i = 0; i < BOARD_SIZE + 1; i++)
      for (int j = 0; j < BOARD_SIZE; j++)
        theBoard[i][j] = -1;
    piece_list = new Piece[PIECE_COUNT];
    move_list = null;
  }

  /**
   * Copy constructor
   *
   * @param b Board
   */
  public Board(Board b) {
    // Create the board
    theBoard = new int[BOARD_SIZE + 1][BOARD_SIZE];
    // Copy the existing board
    for (int i = 0; i < BOARD_SIZE + 1; i++)
      for (int j = 0; j < BOARD_SIZE; j++)
        theBoard[i][j] = b.theBoard[i][j];
    // Copy the pieces
    piece_list = new Piece[PIECE_COUNT];
    piece_count = b.piece_count;
    for (int i = 0; i < b.piece_count; i++)
      piece_list[i] = new Piece(b.piece_list[i]);
    // Copy the moves that got us here
    if (b.move_list != null) {
      Move ptr, b_ptr;
      Move temp = new Move(b.move_list);
      ptr = move_list = temp;
      b_ptr = b.move_list;
      while (b_ptr.next != null) {
        b_ptr = b_ptr.next;
        temp = new Move(b_ptr);
        ptr.next = temp;
        ptr = temp;
      }
    }
    // Need to capture everything because after a move one of the pieces and
    // the board will no longer be the same.
  }

  /**
   * This method imports the board from the string array newBoard into
   * theBoard data structure. The origin of the array is also translated to
   * standard cartesian.
   *
   * @param newBoard String[][]
   */
  public void importBoard(String[][] newBoard) {

    if (SAFARI)
      importSafariBoard(newBoard);
    else
      importRushBoard(newBoard);
  }

  private void importSafariBoard(String[][] newBoard) {
    int i;
    int col, row;
    // For the Board class, x and y origin is bottom left
    // For the newBoard string, x and y origin is upper left
    for (int x = 0; x < BOARD_SIZE; x++)
      for (int y = 0; y < BOARD_SIZE; y++) {
        row = BOARD_INDEX - x;
        col = y;
        // First make sure that this piece is not in the piece list
        if ((i = findPiece(newBoard[x][y])) >= 0) {
          theBoard[col][row] = i;
          continue;
        }
        // This is a new 1x3 (Elephants and Rhinos)
        if (newBoard[x][y].charAt(0) == 'E' || newBoard[x][y].charAt(0) == 'R') {
          // Which direction is this object facing?
          if (x + 1 <= BOARD_INDEX)
            if (newBoard[x + 1][y].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x + 1][y].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 1, 3,
                      Piece.NORTH_SOUTH, newBoard[x][y]); // East West
          if (y + 1 <= BOARD_INDEX)
            if (newBoard[x][y + 1].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x][y + 1].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 3, 1,
                      Piece.EAST_WEST, newBoard[x][y]); // North South
          theBoard[col][row] = piece_count - 1;
        }
        // This is a new 1x2 (Lions, Lionesses(Cheetahs), Zebras, Impalas )
        if (newBoard[x][y].charAt(0) == 'L' || newBoard[x][y].charAt(0) == 'C' ||
                newBoard[x][y].charAt(0) == 'Z' || newBoard[x][y].charAt(0) == 'I') {
          // Which direction is this object facing?
          if (x + 1 <= BOARD_INDEX)
            if (newBoard[x + 1][y].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x + 1][y].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 1, 2,
                      Piece.NORTH_SOUTH, newBoard[x][y]); // East West
          if (y + 1 <= BOARD_INDEX)
            if (newBoard[x][y + 1].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x][y + 1].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 2, 1,
                      Piece.EAST_WEST, newBoard[x][y]); // North South
          theBoard[col][row] = piece_count - 1;
        }
        // This is a new 2x2 (Termites)
        if (newBoard[x][y].charAt(0) == 'T') {
          // Dont' need to worry about direction for termites, they can move both
          piece_list[piece_count++] = new Piece(col, row, 2, 2, Piece.NSEW,
                  newBoard[x][y]);
          theBoard[col][row] = piece_count - 1;
        }
        // This is the new 2x2 Rover
        if (newBoard[x][y].charAt(0) == 'X') {
          // Dont' need to worry about direction for rover, can move both
          piece_list[piece_count++] = new Piece(col, row, 2, 2, Piece.NSEW,
                  newBoard[x][y]);
          theBoard[col][row] = piece_count - 1;
        }
      }
  }

  private void importRushBoard(String[][] newBoard) {
    int i;
    int col, row;
    // For the Board class, x and y origin is bottom left
    // For the newBoard string, x and y origin is upper left
    for (int x = 0; x < BOARD_SIZE; x++)
      for (int y = 0; y < BOARD_SIZE; y++) {
        row = BOARD_INDEX - x;
        col = y;
        // First make sure that this piece is not in the piece list
        if ((i = findPiece(newBoard[x][y])) >= 0) {
          theBoard[col][row] = i;
          continue;
        }
        // This is a new 1x3 (Truck)
        if (newBoard[x][y].charAt(0) >= 'O' && newBoard[x][y].charAt(0) <= 'R') {
          // Which direction is this object facing?
          if (x + 1 <= BOARD_INDEX)
            if (newBoard[x + 1][y].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x + 1][y].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 1, 3,
                      Piece.NORTH_SOUTH, newBoard[x][y]); // North South
          if (y + 1 <= BOARD_INDEX)
            if (newBoard[x][y + 1].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x][y + 1].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 3, 1,
                      Piece.EAST_WEST, newBoard[x][y]); // East West
          theBoard[col][row] = piece_count - 1;
        }
        // This is a new 1x2 (Car)
        if (newBoard[x][y].charAt(0) >= 'A' && newBoard[x][y].charAt(0) <= 'N') {
          // Which direction is this object facing?
          if (x + 1 <= BOARD_INDEX)
            if (newBoard[x + 1][y].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x + 1][y].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 1, 2,
                      Piece.NORTH_SOUTH, newBoard[x][y]); // North South
          if (y + 1 <= BOARD_INDEX)
            if (newBoard[x][y + 1].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x][y + 1].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 2, 1,
                      Piece.EAST_WEST, newBoard[x][y]); // East West
          theBoard[col][row] = piece_count - 1;
        }
        // This is the car
        if (newBoard[x][y].charAt(0) == 'X') {
          // Which direction is this object facing?
          if (x + 1 <= BOARD_INDEX)
            if (newBoard[x + 1][y].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x + 1][y].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 1, 2,
                      Piece.NORTH_SOUTH, newBoard[x][y]); // North South
          if (y + 1 <= BOARD_INDEX)
            if (newBoard[x][y + 1].charAt(0) == newBoard[x][y].charAt(0) &&
                    newBoard[x][y + 1].charAt(1) == newBoard[x][y].charAt(1))
              piece_list[piece_count++] = new Piece(col, row, 2, 1,
                      Piece.EAST_WEST, newBoard[x][y]); // East West
          theBoard[col][row] = piece_count - 1;
        }
      }
  }

  /**
   * Helper function for the importBoard function. Searches the current list
   * of pieces for the piece indicated by the String name and returns it's
   * index.
   *
   * @param name String
   * @return int
   */
  protected int findPiece(String name) {
    for (int i = 0; i < piece_count; i++)
      if (piece_list[i].name.equals(name))
        return i;
    return -1;
  }

  /**
   * Generate the list of possible moves for the current search node
   *
   * @return Move
   */
  Move genMoves() {
    //linked list of all possible moves
    Move result = null;
    int i, j;
    // For each piece on the board
    for (i = 0; i < piece_count; i++) {
      // If it can move North and South (Up/Down)
      if (piece_list[i].move_direction == Piece.NORTH_SOUTH ||
              piece_list[i].move_direction == Piece.NSEW) {
        for (j = 1; j < BOARD_INDEX; j++) {
          // Can we move North 'j' squares?
          // First do a bounds check
          if (piece_list[i].y + piece_list[i].dy + j - 1 > BOARD_INDEX)
            break;
          // Then check for colliding pieces
          if (theBoard[piece_list[i].x][piece_list[i].y + piece_list[i].dy + j - 1] == -1) {
            // Make sure that this isn't a 2x2 piece
            if (piece_list[i].dx != 1)
              if (theBoard[piece_list[i].x + piece_list[i].dx - 1][piece_list[i].y + piece_list[i].dy + j - 1] != -1)
                break;
            Move m = new Move(i, Move.NORTH, j, result);
            result = m;
          } else
            break;
        }
        for (j = 1; j < BOARD_INDEX; j++) {
          // Can we move South 'j' squares?
          // First do a bounds check
          if (piece_list[i].y - j < 0)
            break;
          // Then check for colliding pieces
          if (theBoard[piece_list[i].x][piece_list[i].y - j] == -1) {
            // Make sure that this isn't a 2x2 piece
            if (piece_list[i].dx != 1)
              if (theBoard[piece_list[i].x + piece_list[i].dx - 1][piece_list[i].y - j] != -1)
                break;
            Move m = new Move(i, Move.SOUTH, j, result);
            result = m;
          } else
            break;
        }
      }
      // If it can move East and West (Left/Right)
      if (piece_list[i].move_direction == Piece.EAST_WEST ||
              piece_list[i].move_direction == Piece.NSEW) {
        for (j = 1; j < BOARD_INDEX; j++) {
          // Can we move East 'j' squares?
          // First do a bounds check
          if (piece_list[i].x + piece_list[i].dx + j - 1 > BOARD_INDEX)
            break;
          // Then check for colliding pieces
          if (theBoard[piece_list[i].x + piece_list[i].dx + j - 1][piece_list[i].y] == -1) {
            // Make sure that this isn't a 2x2 piece
            if (piece_list[i].dy != 1)
              if (theBoard[piece_list[i].x + piece_list[i].dx + j - 1][piece_list[i].y + piece_list[i].dy - 1] != -1)
                break;
            Move m = new Move(i, Move.EAST, j, result);
            result = m;
          } else
            break;
        }
        if (piece_list[i].name.equals("X0") && piece_list[i].y == BOARD_EXIT_Y &&
                (piece_list[i].x + piece_list[i].dx) < BOARD_EXIT_X) {
          if (theBoard[piece_list[i].x + piece_list[i].dx + j - 1][piece_list[i].y] == -1 &&
                  theBoard[piece_list[i].x + piece_list[i].dx + j - 1][piece_list[i].y + piece_list[i].dy - 1] == -1) {
            Move m = new Move(i, Move.EAST, j, result);
            result = m;
          }
        }
        for (j = 1; j < BOARD_INDEX; j++) {
          // Can we move West 'j' squares?
          // First do a bounds check
          if (piece_list[i].x - j < 0)
            break;
          // Then check for colliding pieces
          if (theBoard[piece_list[i].x - j][piece_list[i].y] == -1) {
            // Make sure that this isn't a 2x2 piece
            if (piece_list[i].dy != 1)
              if (theBoard[piece_list[i].x - j][piece_list[i].y + piece_list[i].dy - 1] != -1)
                break;
            Move m = new Move(i, Move.WEST, j, result);
            result = m;
          } else
            break;
        }
      }
    }
    // Return linked list of possible moves from node
    return result;
  }

  /**
   * make sent move
   *
   * @param m Move
   */
  public void makeMove(Move m) {
    int x, y;

    // First remove the piece from theBoard in its current location.
    x = piece_list[m.piece_index].x;
    y = piece_list[m.piece_index].y;
    for (int i = 0; i < piece_list[m.piece_index].dx; i++)
      for (int j = 0; j < piece_list[m.piece_index].dy; j++)
        theBoard[x + i][y + j] = -1;

    if (m.direction == Move.NORTH) {
      piece_list[m.piece_index].y += m.spaces;
    } else if (m.direction == Move.SOUTH) {
      piece_list[m.piece_index].y -= m.spaces;
    } else if (m.direction == Move.EAST) {
      piece_list[m.piece_index].x += m.spaces;
    } else { // Move.WEST
      piece_list[m.piece_index].x -= m.spaces;
    }

    // Now place the piece in theBoard in it's new location.
    x = piece_list[m.piece_index].x;
    y = piece_list[m.piece_index].y;
    for (int i = 0; i < piece_list[m.piece_index].dx; i++)
      for (int j = 0; j < piece_list[m.piece_index].dy; j++) {
        theBoard[x + i][y + j] = m.piece_index;
      }

    Move temp = new Move(m);
    temp.next = move_list;
    move_list = temp;
  }

  /**
   * reverse move
   *
   * @param m Move
   */
  public void reverseMove(Move m) {
    int x, y;

    // First remove the piece from theBoard in its current location.
    x = piece_list[m.piece_index].x;
    y = piece_list[m.piece_index].y;
    for (int i = 0; i < piece_list[m.piece_index].dx; i++)
      for (int j = 0; j < piece_list[m.piece_index].dy; j++)
        theBoard[x + i][y + j] = -1;

    if (m.direction == Move.NORTH) {
      piece_list[m.piece_index].y -= m.spaces;
    } else if (m.direction == Move.SOUTH) {
      piece_list[m.piece_index].y += m.spaces;
    } else if (m.direction == Move.EAST) {
      piece_list[m.piece_index].x -= m.spaces;
    } else { // Move.WEST
      piece_list[m.piece_index].x += m.spaces;
    }

    // Now place the piece in theBoard in it's new location.
    x = piece_list[m.piece_index].x;
    y = piece_list[m.piece_index].y;
    for (int i = 0; i < piece_list[m.piece_index].dx; i++)
      for (int j = 0; j < piece_list[m.piece_index].dy; j++) {
        theBoard[x + i][y + j] = m.piece_index;
      }

    // take the last move off the move_list
    move_list = move_list.next;
  }

  public boolean goalCheck(Board b) {
    int x = b.findPiece("X0");
    Piece redPiece = b.piece_list[x];
    if (redPiece.x == BOARD_EXIT_X-1 && redPiece.y == BOARD_EXIT_Y) {
      return true;
    } else {
      return false;
    }
  }

  /**
   * Equals method
   * Compares the 2 theBoard fields for equality
   * @param o the object being compared
   * @return equals or not
   */
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Board board = (Board) o;
    return Arrays.deepEquals(theBoard, board.theBoard);
  }
}


