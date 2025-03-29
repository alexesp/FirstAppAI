using TorchSharp;

int[,] maze1 = {
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 }, //row 0
    { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 }, //row 1
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 2
    { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 3
    { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 4
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 5
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 6
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 7
    { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 8
    { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 9
    { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 10
    { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }  //row 11 (start position is (11, 5))
};


const string UP = "up";
const string DOWN = "down";
const string LEFT = "left";
const string RIGHT = "right";

string[] actions = { UP, DOWN, LEFT, RIGHT };

int[,] rewards;

const int WALL_REWARD_VALUE = -500;
const int FLOOR_REWARD_VALUE = -10;
const int GOAL_REWARD_VALUE = 500;

void setupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);

    rewards = new int[mazeRows, mazeColumns];

    for (int i = 0; i < mazeRows; i++)
    {
        for (int j = 0; j < mazeColumns; j++)
        {
            switch (maze[i, j])
            {
                case 0:
                    rewards[i, j] = wallValue;
                    break;
                case 1:
                    rewards[i, j] = floorValue;
                    break;
                case 2:
                    rewards[i, j] = goalValue;
                    break;
            }
        }
    }
}

torch.Tensor qValues;

void setupQValues(int[,] maze)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);
    qValues = torch.zeros(mazeRows, mazeColumns, 4);

}

bool hasHitWallOrEndOfMaze(int currentRow, int currentColumn, int floorValue)
{
    return rewards[currentRow, currentColumn] != floorValue;
}

long determineNextAction(int currentRow, int currentColum, float epsilon)
{
    Random random = new Random();
    double randomBetween0and1 = random.NextDouble();

    long nextAction = randomBetween0and1 < epsilon ? torch.argmax(qValues[currentRow, currentColum]).item<long>() : random.Next(4);

    return nextAction;
}

(int, int) moveOneSpace(int[,] maze, int currentRow, int currentColumn, long currentAction)
{
    int mazeRows = maze.GetLength(0);
    int mazeColumns = maze.GetLength(1);

    int nextRow = currentRow;
    int nextColumn = currentColumn;

    if (actions[currentAction] == UP && currentRow > 0)
    {
        nextRow--;

    }
    else if (actions[currentAction] == DOWN && currentRow < mazeRows - 1)
    {
        nextRow++;
    }
    else if (actions[currentAction] == LEFT && currentColumn > 0)
    {
        nextColumn--;
    }
    else if (actions[currentAction] == RIGHT && currentColumn < mazeColumns - 1)
    {
        nextColumn++;
    }
    return (nextRow, nextColumn);
}

void trainTheModel(int[,] maze, int floorValue, float epsilon, float discountFactor, float learningRate, float episodes)
{
    for(int episode = 0;episode < episodes;episode++)
    {
        Console.WriteLine("-----Starting episode " + episode + "-----");
        int currentRow = 11;
        int currentColumn = 5;

        while(!hasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            long currentAction = determineNextAction(currentRow, currentColumn, epsilon);
            int previosRow = currentRow;
            int previosColumn = currentColumn;
            (int, int) nextMove = moveOneSpace(maze, previosRow, previosColumn, currentAction);
            currentRow = nextMove.Item2;
            currentColumn = nextMove.Item2;
            float reward = rewards[currentRow, currentColumn];
            float previousQValue = qValues[previosRow, previosColumn, currentAction].item<float>();
            float temporalDifference = reward + (discountFactor * torch.max(qValues[currentRow, currentColumn])).item<float>();
            float nextQValue = previousQValue + (learningRate * temporalDifference);
            qValues[currentRow, currentColumn, currentAction] = nextQValue;
        }

        Console.WriteLine("----Finished episode " + episode + "----");
    }

    Console.WriteLine("Completed training!");
}