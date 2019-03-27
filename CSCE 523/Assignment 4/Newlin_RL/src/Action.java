public class Action {

    int x;
    int y;
    double reward;

    public Action(int x, int y, double reward) {
        setX(x);
        setY(y);
        setReward(reward);
    }

    public Action() {
        this(0,0,0);
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public double getReward() {
        return reward;
    }

    public void setReward(double reward) {
        this.reward = reward;
    }
}
