from gomoku_env import GomokuEnvironment


def test_gomoku_environment():
    env = GomokuEnvironment(board_size=15, config_path="rewards/rewards_2.yml")
    env.reset()

    # Case 1: Invalid move
    print("Test Case 1: Invalid Move")
    env.board[5, 5] = 1  
    board, reward, done, info = env.step((5, 5))  
    print(board, reward, info)  
    assert reward == 0, "Test Case 1 Failed: Invalid move did not return 0"

    # Case 2: Forming 2 in a row
    print("\nTest Case 2: Forming 2 in a Row")
    env.reset()
    board, reward, done, info = env.step((7, 7))  # X
    board, reward, done, info = env.step((0, 0))  # O
    board, reward, done, info = env.step((7, 8))  # X (forms two in a row)
    print(board, reward, "X")  # Expected: 0.1
    assert reward == 0.1, "Test Case 2 Failed: Reward for forming 2 in a row is incorrect"

    # Case 3: Forming 3 in a row
    print("\nTest Case 3: Forming 3 in a Row")
    env.reset()
    board, reward, done, info = env.step((7, 7))  # X
    board, reward, done, info = env.step((0, 0))  # O
    board, reward, done, info = env.step((7, 8))  # X
    board, reward, done, info = env.step((0, 1))  # O
    board, reward, done, info = env.step((7, 9))  # X (forms three in a row)
    print(board, reward, "X")  # Expected: 0.2
    assert reward == 0.2, "Test Case 3 Failed: Reward for forming 3 in a row is incorrect"

    # Case 4: Forming 4 in a row
    print("\nTest Case 4: Forming 4 in a Row")
    env.reset()
    board, reward, done, info = env.step((7, 7))  # X
    board, reward, done, info = env.step((0, 0))  # O
    board, reward, done, info = env.step((7, 8))  # X
    board, reward, done, info = env.step((0, 1))  # O
    board, reward, done, info = env.step((7, 9))  # X
    board, reward, done, info = env.step((7, 6))  # O
    board, reward, done, info = env.step((7, 10))  # X (forms four in a row)
    print(board, reward, "X")  # Expected: 0.5
    assert reward == 0.5, "Test Case 4 Failed: Reward for forming 4 in a row is incorrect"

    # Case 5: Creating a double threat
    print("\nTest Case 5: Creating a Double Threat")
    env.reset()
    board, reward, done, info = env.step((7, 7))  # X
    board, reward, done, info = env.step((0, 0))  # O
    board, reward, done, info = env.step((7, 8))  # X
    board, reward, done, info = env.step((0, 1))  # O
    board, reward, done, info = env.step((7, 9))  # X
    board, reward, done, info = env.step((0, 2))  # O
    board, reward, done, info = env.step((7, 10))  # X (forms four in a row)
    print(board, reward, "X")  # Expected: 0.8
    assert reward == 0.8, "Test Case 5 Failed: Reward for creating a double threat is incorrect"


    # Case 6: Placing far from current group
    print("\nTest Case 6: Placing Far from Current Group")
    env.reset()
    board, reward, done, info = env.step((7, 7))  # X
    board, reward, done, info = env.step((0, 0))  # O
    board, reward, done, info = env.step((7, 8))  # X
    board, reward, done, info = env.step((0, 1))  # O
    board, reward, done, info = env.step((7, 2))  # X
    print(board, reward, "X")  # Expected: -0.3
    assert reward == -0.3, "Test Case 6 Failed: Penalty for placing far from the group is incorrect"

    # Case 7: Winning the game
    print("\nTest Case 7: Winning the Game")
    env.reset()
    env.board[7, 7] = 1
    env.board[7, 8] = 1
    env.board[7, 9] = 1
    env.board[7, 10] = 1
    board, reward, done, info = env.step((7, 11))  # X (wins the game)
    print(board, reward, done, info)  # Expected: 1.0, game ends
    assert reward == 1.0, "Test Case 7 Failed: Winning the game did not return correct reward"
    assert done is True, "Test Case 7 Failed: Winning the game did not end the game"

    # Case 8: Drawing the game
    print("\nTest Case 8: Drawing the Game")
    env = GomokuEnvironment(board_size=3, config_path="rewards/rewards_2.yml")
    env.reset()
    # Fill the board alternately with 1 and 2 to ensure no player wins
    env.board[0, 0] = 1  # X
    env.board[0, 1] = 2  # O
    env.board[0, 2] = 1  # X
    env.board[1, 0] = 2  # O
    env.board[1, 1] = 1  # X
    env.board[1, 2] = 2  # O
    env.board[2, 0] = 1  # X
    env.board[2, 1] = 2  # O
    board, reward, done, info = env.step((2, 2))  # X fills the last spot
    print(board, reward, done, info)  # Expected: 0.0 (draw)
    assert reward == 0.0, "Test Case 8 Failed: Drawing the game did not return correct reward"
    assert done is True, "Test Case 8 Failed: Drawing the game did not end the game"

    # Case 9: Combination of forming multiple rows
    env = GomokuEnvironment(board_size=15, config_path="rewards/rewards_2.yml")
    print("\nTest Case 9: Forming Multiple Rows")
    env.reset()

    # Setup: Create an intersection where X forms horizontal and vertical sequences
    env.board[7, 6] = 1  # Horizontal: X
    env.board[7, 8] = 1  # Horizontal: X
    env.board[6, 7] = 1  # Vertical: X
    env.board[8, 7] = 1  # Vertical: X

    board, reward, done, info = env.step((7, 7))  # X extends horizontal row to 3
    print(board, reward, "X")  # Expected: Sum of rewards for 3-in-a-row (horizontal) + 3-in-a-row (vertical)
    assert reward == 0.4, "Test Case 9 Failed: Reward for forming two 3 in a rows is incorrect"
    

    board, reward, done, info = env.step((0, 0))  # O random move
    print(board, reward, "O")  # Expected: Default reward (0)
    assert reward == 0, "Test Case 9 Failed: Reward for random move is incorrect"

    # Another combination move
    board, reward, done, info = env.step((9, 7))  # X extends vertical row to 4
    print(board, reward, "X")  # Expected: Double threat
    assert reward == 0.8, "Test Case 9 Failed: Reward for double threat is incorrect"

    board, reward, done, info = env.step((0, 1))  # O random move
    print(board, reward, "O")  
    assert reward == 0.1, "Test Case 9 Failed: Reward for 2 in a row is incorrect"

    board, reward, done, info = env.step((12, 12))  
    print(board, reward, "X")  
    assert reward == -0.3, "Test Case 9 Failed: Reward for placing a stone too far incorrect"

    # Case 10
    print("\nTest Case 10: Blocking Multiple Rows")
    env.reset()

    # Opponent setup: Two threats, one horizontal and one vertical
    env.board[7, 6] = 2  # O
    env.board[7, 7] = 2  # O
    env.board[7, 8] = 2  # O
    env.board[7, 9] = 2  # O
    env.board[6, 10] = 2  # O
    env.board[5, 10] = 2  # O
    env.board[4, 10] = 2  # O

    # Player blocks both threats
    board, reward, done, info = env.step((7, 10))  # X
    print(board, reward, "X")  # Expected: Sum of blocking 3 (0.5 for horizontal + 0.2 for vertical) = 0.7
    assert reward == 0.7, "Test Case 10 Failed: Reward for blocking 2 threats is incorrect"

    # Case 11
    print("\nTest Case 11: Blocking Multiple Rows")
    env.reset()

    # Opponent setup: Two threats, one horizontal and one vertical
    env.board[7, 6] = 2  # O
    env.board[7, 7] = 2  # O
    env.board[7, 8] = 2  # O
    env.board[7, 9] = 2  # O
    env.board[6, 10] = 2  # O
    env.board[5, 10] = 2  # O
    env.board[4, 10] = 2  # O
    env.board[3, 10] = 2  # O

    # Player blocks both threats
    board, reward, done, info = env.step((7, 10))  # X
    print(board, reward, "X")  # Expected: Sum of blocking 3 (0.5 for horizontal + 0.5 for vertical) = 1
    assert reward == 1, "Test Case 11 Failed: Reward for blocking 2 threats is incorrect"

    # Case 12
    print("\nTest Case 12: Blocking and Forming Rows")
    env.reset()

    # Opponent setup: One horizontal threat
    env.board[7, 6] = 2  # O
    env.board[7, 7] = 2  # O
    env.board[7, 8] = 2  # O
    env.board[5, 9] = 2  # O

    # Player setup: Three vertical stones
    env.board[6, 9] = 1  # X
    env.board[8, 9] = 1  # X
    env.board[9, 9] = 1  # X

    # Player blocks the opponent while forming 4-in-a-row
    board, reward, done, info = env.step((7, 9))  # X
    print(board, reward, "X")  # Expected: Reward for blocking 3 (0.2) + forming 4-in-a-row (0.5) = 0.7
    assert reward == 0.7, "Test Case 12 Failed: Reward for blocking 3 and forming 4-in-a-row is incorrect"

    # Case 13
    print("\nTest Case 13: Blocking One Row")
    env.reset()

    # Opponent setup: Two threats, one horizontal and one vertical
    env.board[7, 6] = 2  # O
    env.board[7, 7] = 2  # O
    env.board[7, 8] = 2  # O
    env.board[7, 9] = 2  # O

    # Player blocks both threats
    board, reward, done, info = env.step((7, 10))  # X
    print(board, reward, "X")  # Expected: 0.5
    assert reward == 0.5, "Test Case 12 Failed: Reward for blocking 4 is incorrect"

    # Case 14: First move for X
    print("\nTest Case 14: First move for X")
    env.reset()
    board, reward, done, info = env.step((5, 5))  
    print(board, reward, "X")  # Expected: 0
    assert reward == 0, "Test Case 13 Failed: Invalid reward for the first X"

    # Case 15: First move for O
    print("\nTest Case 15: First move for X")
    env.reset()
    env.step((5, 5))  
    board, reward, done, info = env.step((14, 14))  
    print(board, reward, "O")  # Expected: 0
    assert reward == 0, "Test Case 15 Failed: Invalid reward for the first O"


    # Case 16: Stone too far placed 
    print("\nTest Case 15: Stone too far placed ")
    env.reset()
    env.step((0, 1))  # X
    env.step((0, 5))   # O
    board, reward, done, info = env.step((9, 9))  
    print(board, reward, "X")  # Expected: -0.3
    assert reward == -0.3, "Test Case 16 Failed: Invalid reward for stone placed too far"



test_gomoku_environment()