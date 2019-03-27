import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Vector;
import java.util.concurrent.ExecutionException;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.JPanel;
import javax.swing.SwingWorker;

public class CustomPanel extends JPanel {

	private static final long serialVersionUID = 1L;

	Graphics offscreen;     // Declaration of offscreen buffer
	BufferedImage image;    // Image associated with the buffer
	private Image background, black_piece, white_piece; // The artwork

	public int user_move;   // State variable for user player move selection

	// Player move selection states
	public static final int NOT_MOVE = 1000;
	public static final int PICK_PIECE = 1001;
	public static final int PICK_MOVE = 1002;
	public static final int END_MOVE = 1003;
	public static final int WAIT1 = 1004;
	public static final int WAIT2 = 1005;

	WorkBoard board = new WorkBoard(); // The LOA board
	public int x1, y1, x2, y2; // User move information

	private int depth;
	private Move lastmove = null;
	private LOAGUI lgui;
	LOAWorker worker;
	private String player;
	private String computer;

	// Set SELF_PLAY to true to play by yourself (also, always select black when you start)
	private Boolean SELF_PLAY = false;
	
	/**
	 * The LOAWorker generates a thread that calls the WorkBoard bestMove
	 * method.
	 */
	final class LOAWorker extends SwingWorker<Integer, Void> {

		protected Integer doInBackground() throws Exception {
			long startTime;

			startTime = System.currentTimeMillis();
			board.bestMove(getDepth());
			startTime = System.currentTimeMillis() - startTime;
			int result = board.try_move(board.best_move);
			lgui.statusTextArea.append(computer + " Move: " + board.best_move.name()
					+ "\n");
			lgui.statusTextArea.append("Time: " + (float) (startTime) / 1000.0
					+ " s\n");
			System.out.println("Search Time: " + (startTime) / 1000.0
					+ "s  Best move: " + board.best_move.name() + "\n");
			lgui.status.setText("Your move as " + getPlayer() + ".");
			lastmove = board.best_move;

			return new Integer(result);
		}

		// Can safely update the GUI from this method.
		protected void done() {

			Integer result;
			try {
				// Retrieve the return value of doInBackground.
				result = get();
				if (result == Board.GAME_OVER) {
					if (board.referee() == Board.PLAYER_BLACK) {
						lgui.status.setText("GAME OVER Black wins!");
						lgui.statusTextArea.append("Black wins!");
					} else {
						lgui.status.setText("GAME OVER White wins!");
						lgui.statusTextArea.append("White wins!");
					}
				} else
					user_move = WAIT1;
				repaint();
			} catch (InterruptedException e) {
				// This is thrown if the thread is interrupted.
			} catch (ExecutionException e) {
				// This is thrown if we throw an exception
				// from doInBackground.
			}
		}
	}

	public CustomPanel() {

		setBorder(BorderFactory.createLineBorder(Color.black));

		addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent e) {
				handleMouse(e);
			}

			public void mouseClicked(MouseEvent e) {
				handleMouse(e);
			}
		});
		// Load the artwork
		loadImages();

		setDepth(3);
	}

	public CustomPanel(LOAGUI L) {
		lgui = L;
		setBorder(BorderFactory.createLineBorder(Color.black));

		addMouseListener(new MouseAdapter() {
			public void mouseClicked(MouseEvent e) {
				handleMouse(e);
				validate();
			}
		});
		// Load the artwork
		loadImages();

		setDepth(3);
	}

	protected void handleMouse(MouseEvent e) {
		int grid_x, grid_y;
		Boolean break_flag = false;
		Vector<Move> v;
		int stat;
		while (user_move != NOT_MOVE || break_flag) {
			int x = e.getX();
			int y = e.getY();
			grid_x = (int) ((x - 8) / 35);
			grid_y = (7 - (int) ((y - 8) / 35));

			switch (user_move) {

			case WAIT1:
				// If the user did NOT click a piece return
				if (!board.piece(grid_x, grid_y)) {
					return;
				}
				// If the user picked a piece identify it and fall through to
				// the PICK_PIECE block.
				x1 = grid_x;
				y1 = grid_y;
				user_move = PICK_PIECE;

			case PICK_PIECE:
				v = board.genMoves2(x1, y1);
				if (v.size() == 0)
					user_move = WAIT1;
				else
					user_move = WAIT2;
				repaint();
				return;

			case WAIT2:
				if (board.piece(grid_x, grid_y)) {
					x1 = grid_x;
					y1 = grid_y;
					x2 = y2 = -1;
					user_move = PICK_PIECE;
					break;
				}
				x2 = grid_x;
				y2 = grid_y;
				user_move = PICK_MOVE;

			case PICK_MOVE:
				v = board.genMoves2(x1, y1);
				Move m;
				for (int i = 0; i < v.size(); i++) {
					m = (Move) v.elementAt(i);
					if (m.x2 == x2 && m.y2 == y2) {
						// valid move, need to move piece and update screen.
						stat = board.try_move(m);
						user_move = NOT_MOVE;
						lgui.statusTextArea.append(player + " Move: " + m.name()
								+ "\n");
						lgui.status.setText("Computer's move as " + computer + ".");
						lastmove = m;
						if (stat == Board.GAME_OVER) {
							if (board.referee() == Board.PLAYER_BLACK) {
								lgui.status.setText("GAME OVER Black wins!");
								lgui.statusTextArea.append("Black wins!\n");
							} else {
								lgui.status.setText("GAME OVER White wins!");
								lgui.statusTextArea.append("White wins!\n");
							}
							repaint();
							return;
						}
						repaint();
						if (SELF_PLAY)
							user_move = NOT_MOVE;
						else {
							worker = new LOAWorker();
							worker.execute();
						}
						return;
					} 
				}
				if (board.piece(grid_x, grid_y)) {
					// they selected another piece
					user_move = PICK_PIECE;
					break;
				} else {
					// they must have clicked a random location
					return;
				}
			default:
				repaint();
			}
		}
		if (SELF_PLAY)
			user_move = WAIT1;
	}

	public void runWorkerExtern() {
		worker = new LOAWorker();
		worker.execute();
	}
	
	/**
	 * Load the artwork and initialize the drawing surfaces
	 */
	public void loadImages() {
		try {
			background = ImageIO.read(new File("src/LOA-Grid.png"));
			black_piece = ImageIO.read(new File("src/LOA-Black.png"));
			white_piece = ImageIO.read(new File("src/LOA-White.png"));
		} catch (IOException ex) {
			// handle exception...
		}
		// allocation of offscreen buffer
		image = new BufferedImage(300, 300, BufferedImage.TYPE_INT_RGB);
		offscreen = image.getGraphics();
		// initialize game state
		user_move = NOT_MOVE;
	}

	public Dimension getPreferredSize() {
		return new Dimension(300, 300);
	}

	/**
	 * The overridden paint function, copies the background and all of the other
	 * graphics bits to the background Graphic that will be updated when we call
	 * canvas.repaint at the end.
	 * 
	 * @param g
	 *            Graphics the canvas to draw the board on.
	 */
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		// showStatus( status );

		// Copy the background image
		offscreen.drawImage(background, 0, 0, 300, 300, this);

		// If the computer moved previously show this move.
		// Doing this first so that when we draw the pieces, it overwrites part
		// of
		// line.
		if (lastmove != null) {
			offscreen.setColor(Color.yellow);
			offscreen.drawLine(lastmove.x1 * 35 + 25,
					(7 - lastmove.y1) * 35 + 25, lastmove.x2 * 35 + 25,
					(7 - lastmove.y2) * 35 + 25);
		}

		// Place each piece in the correct location.
		for (int x = 0; x < 8; x++)
			for (int y = 0; y < 8; y++) {
				if (board.checker_of(Board.BLACK_CHECKER, x, y))
					offscreen.drawImage(black_piece, (x) * 35 + 11,
							(7 - y) * 35 + 11, 30, 30, this);
				if (board.checker_of(Board.WHITE_CHECKER, x, y))
					offscreen.drawImage(white_piece, (x) * 35 + 11,
							(7 - y) * 35 + 11, 30, 30, this);
			}
		// If the player is moving, show him his possible moves.
		if (user_move == WAIT2) {
			Vector<Move> v = board.genMoves2(x1, y1);
			Move m;
			offscreen.setColor(Color.red);
			for (int i = 0; i < v.size(); i++) {
				m = (Move) v.elementAt(i);
				offscreen.drawLine(x1 * 35 + 25, (7 - y1) * 35 + 25,
						m.x2 * 35 + 25, (7 - m.y2) * 35 + 25);
			}
		}

		g.drawImage(image, 0, 0, null);
	}

	public int getDepth() {
		return depth;
	}

	public void setDepth(int depth) {
		this.depth = depth;
	}

	public String getPlayer() {
		return player;
	}

	public void setPlayer(String player) {
		this.player = player;
	}
	
	public String getComputer() {
		return computer;
	}

	public void setComputer(String computer) {
		this.computer = computer;
	}
}
