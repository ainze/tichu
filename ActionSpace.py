import numpy as np


def two_cards_indices():
    print("two_cards_indices")
    # Step 1: Initialize action_indices as a 13 × 13 array of zeros
    action_indices = np.zeros((14, 14), dtype=int)

    # Step 2: Initialize inverse_indices2 as a 33 × 3 array of zeros
    inverse_indices2 = np.zeros((36, 2), dtype=int)

    # Step 3: Initialize i = 0
    i = 0

    # Step 4: for c1 = 0 to 11 do
    for c1 in range(13):
        # Step 5: n1 = min(c1 + 3, 12)
        spread = min(c1 + 3, 13)
        # Step 6: for c2 = c1 + 1 to n1 do
        for c2 in range(c1 + 1, spread + 1):
            # Step 7: action_indices[c1, c2] = i
            action_indices[c1, c2] = i
            # Step 8: inverse_indices2[i, :] = [c1, c2]
            inverse_indices2[i, :] = [c1, c2]
            # Step 9: i += 1
            i += 1

    print("action_indices:")
    print(action_indices)
    print("\ninverse_indices2:")
    print(inverse_indices2)


def two_cards_phoenix_indices():
    print("two_cards_phoenix_indices")

    # Step 1: Initialize action_indices as a 13 × 13 array of zeros
    action_indices = np.zeros((14, 14), dtype=int)

    # Step 2: Initialize inverse_indices2 as a 33 × 3 array of zeros
    inverse_indices2 = np.zeros((13, 2), dtype=int)

    # Step 3: Initialize i = 0
    i = 0

    # Step 4: for c1 = 0 to 11 do
    c1 = 0

    # Step 6: for c2 = c1 + 1 to n1 do
    for c2 in range(c1 + 1, 14):
        # Step 7: action_indices[c1, c2] = i
        action_indices[c1, c2] = i
        # Step 8: inverse_indices2[i, :] = [c1, c2]
        inverse_indices2[i, :] = [c1, c2]
        # Step 9: i += 1
        i += 1

    print("action_indices:")
    print(action_indices)
    print("\ninverse_indices2:")
    print(inverse_indices2)


def three_cards_indices():
    print("three_cards_indices")

    action_indices = np.zeros((14, 14, 14), dtype=int)

    inverse_indices = np.zeros((34, 3), dtype=int)

    i = 0

    for c1 in range(12):
        n1 = min(c1 + 2, 13)
        for c2 in range(c1 + 1, n1 + 1):
            n2 = min(c1 + 3, 13)
            for c3 in range(c2 + 1, n2 + 1):
                action_indices[c1, c2, c3] = i
                inverse_indices[i, :] = [c1, c2, c3]
                i += 1

    print("action_indices:")
    print(action_indices)
    print("\ninverse_indices:")
    print(inverse_indices)


def three_cards_phoenix_indices():
    # same as two card, but you assume first card is phoenix
    print("three_cards_phoenix_indices")
    # Step 1: Initialize action_indices as a 13 × 13 array of zeros
    action_indices = np.zeros((14, 14), dtype=int)

    # Step 2: Initialize inverse_indices2 as a 33 × 3 array of zeros
    inverse_indices = np.zeros((36, 3), dtype=int)

    # Step 3: Initialize i = 0
    i = 0

    # Step 4: for c1 = 0 to 11 do
    for c1 in range(13):
        # Step 5: n1 = min(c1 + 3, 12)
        spread = min(c1 + 3, 13)
        # Step 6: for c2 = c1 + 1 to n1 do
        for c2 in range(c1 + 1, spread + 1):
            # Step 7: action_indices[c1, c2] = i
            action_indices[c1, c2] = i
            # Step 8: inverse_indices2[i, :] = [c1, c2]
            inverse_indices[i, :] = [0, c1, c2]
            # Step 9: i += 1
            i += 1

    print("action_indices:")
    print(action_indices)
    print("\ninverse_indices:")
    print(inverse_indices)

def four_card_bomb():
    print("four_card_bomb")

    action_indices = np.zeros((14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((11, 4), dtype=int)

    i = 0

    for c1 in range(11):
        c2 = c1 + 1
        c3 = c1 + 2
        c4 = c1 + 3
        action_indices[c1, c2, c3, c4] = i
        inverse_indices[i, :] = [c1, c2, c3, c4]
        i += 1

    print("action_indices:")
    print(action_indices)
    print("\ninverse_indices:")
    print(inverse_indices)

if __name__ == "__main__":
    two_cards_indices()
    two_cards_phoenix_indices()
    three_cards_indices()
    three_cards_phoenix_indices()
    four_card_bomb()
