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

    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices2:")
    # print(inverse_indices2)

    return inverse_indices2


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

    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices2:")
    # print(inverse_indices2)

    return inverse_indices2


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

    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices


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

    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices


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

    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices


def two_ladder_indices():
    print("two_ladder_indices")

    action_indices = np.zeros((14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((280, 4), dtype=int)

    i = 0

    for c1 in range(11):
        n1 = min(c1 + 5, 13)
        for c2 in range(c1 + 1, n1 + 1):
            n2 = min(c1 + 6, 13)
            for c3 in range(c2 + 1, n2 + 1):
                n3 = min(c1 + 7, 13)
                for c4 in range(c3 + 1, n3 + 1):
                    #print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
                    action_indices[c1, c2, c3, c4] = i
                    inverse_indices[i, :] = [c1, c2, c3, c4]
                    i += 1

    # print(f'i: {i}')
    #print("action_indices:")
    #print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices


def two_ladder_phoenix_indices():
    print("two_ladder_phoenix_indices")

    action_indices = np.zeros((14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((182, 4), dtype=int)

    i = 0

    c0 = 1
    for c1 in range(12):
        n1 = min(c1 + 6, 13)
        for c2 in range(c1 + 1, n1 + 1):
            n2 = min(c1 + 7, 13)
            for c3 in range(c2 + 1, n2 + 1):
                action_indices[c0, c1, c2, c3] = i
                inverse_indices[i, :] = [c0, c1, c2, c3]
                i += 1

    # print(f'i: {i}')
    #print("action_indices:")
    #print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices



def full_house_indices():
    print("full_house_indices")

    action_indices = np.zeros((14, 14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((266, 5), dtype=int)

    i = 0

    for c1 in range(10):
        n1 = min(c1 + 4, 13)
        for c2 in range(c1 + 1, n1 + 1):
            n2 = min(c1 + 5, 13)
            for c3 in range(c2 + 1, n2 + 1):
                n3 = min(c1 + 6, 13)
                for c4 in range(c3 + 1, n3 + 1):
                    n4 = min(c1 + 7, 13)
                    for c5 in range(c4 + 1, n4 + 1):
                        # print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
                        action_indices[c1, c2, c3, c4, c5] = i
                        inverse_indices[i, :] = [c1, c2, c3, c4, c5]
                        i += 1

    # print(f'i: {i}')
    # # print("action_indices:")
    # # print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices



def full_house_phoenix_indices():
    print("full_house_phoenix_indices")

    action_indices = np.zeros((14, 14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((175, 5), dtype=int)

    i = 0

    for c1 in range(19):
        n1 = min(c1 + 4, 13)
        for c2 in range(c1 + 1, n1 + 1):
            n2 = min(c1 + 5, 13)
            for c3 in range(c2 + 1, n2 + 1):
                n3 = min(c1 + 6, 13)
                for c4 in range(c3 + 1, n3 + 1):
                    n4 = min(c1 + 7, 13)
                    #for c5 in range(c4 + 1, n4 + 1):
                    # print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
                    action_indices[0, c1, c2, c3, c4] = i
                    inverse_indices[i, :] = [0, c1, c2, c3, c4]
                    i += 1

    # print(f'i: {i}')
    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices


def six_cards_indices():
    print("six_cards_indices")

    action_indices = np.zeros((14, 14, 14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((1716, 6), dtype=int)

    i = 0

    for c1 in range(8):
        #n1 = min(c1 + 4, 13)
        for c2 in range(c1 + 1, 13):
            for c3 in range(c2 + 1, 13):
                for c4 in range(c3 + 1, 13):
                    for c5 in range(c4 + 1, 13):
                        for c6 in range(c5 + 1, 13):
                            # print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
                            action_indices[c1, c2, c3, c4, c5, c6] = i
                            # inverse_indices[i, :] = [c1, c2, c3, c4, c5]
                            i += 1

    # print(f'i: {i}')
    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices


def seven_cards_indices():
    print("seven_cards_indices")

    action_indices = np.zeros((14, 14, 14, 14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((1716, 7), dtype=int)

    i = 0

    for c1 in range(7):
        #n1 = min(c1 + 4, 13)
        for c2 in range(c1 + 1, 13):
            for c3 in range(c2 + 1, 13):
                for c4 in range(c3 + 1, 13):
                    for c5 in range(c4 + 1, 13):
                        for c6 in range(c5 + 1, 13):
                            for c7 in range(c6 + 1, 13):
                                # print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
                                action_indices[c1, c2, c3, c4, c5, c6, c7] = i
                                # inverse_indices[i, :] = [c1, c2, c3, c4, c5]
                                i += 1

    # print(f'i: {i}')
    # print("action_indices:")
    # print(action_indices)
    # print("\ninverse_indices:")
    # print(inverse_indices)

    return inverse_indices


def eight_cards_indices():
    print("eight_cards_indices")

    action_indices = np.zeros((14, 14, 14, 14, 14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((1716, 8), dtype=int)

    i = 0

    for c1 in range(7):
        #n1 = min(c1 + 4, 13)
        for c2 in range(c1 + 1, 13):
            for c3 in range(c2 + 1, 13):
                for c4 in range(c3 + 1, 13):
                    for c5 in range(c4 + 1, 13):
                        for c6 in range(c5 + 1, 13):
                            for c7 in range(c6 + 1, 13):
                                for c8 in range(c7 + 1, 13):
                                    # print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
                                    action_indices[c1, c2, c3, c4, c5, c6, c7, c8] = i
                                    # inverse_indices[i, :] = [c1, c2, c3, c4, c5]
                                    i += 1

    print(f'i: {i}')
    # print("action_indices:")
    # print(action_indices)
    print("\ninverse_indices:")
    print(inverse_indices)

    return inverse_indices


def nine_cards_indices():
    print("nine_cards_indices")

    action_indices = np.zeros((14, 14, 14, 14, 14, 14, 14, 14, 14), dtype=int)

    inverse_indices = np.zeros((1716, 9), dtype=int)

    i = 0

    for c1 in range(7):
        #n1 = min(c1 + 4, 13)
        for c2 in range(c1 + 1, 13):
            for c3 in range(c2 + 1, 13):
                for c4 in range(c3 + 1, 13):
                    for c5 in range(c4 + 1, 13):
                        for c6 in range(c5 + 1, 13):
                            for c7 in range(c6 + 1, 13):
                                for c8 in range(c7 + 1, 13):
                                    for c9 in range(c8 + 1, 13):
                                        # print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
                                        action_indices[c1, c2, c3, c4, c5, c6, c7, c8, c9] = i
                                        # inverse_indices[i, :] = [c1, c2, c3, c4, c5]
                                        i += 1

    print(f'i: {i}')
    # print("action_indices:")
    # print(action_indices)
    print("\ninverse_indices:")
    print(inverse_indices)

    return inverse_indices


    # def ten_cards_indices():
    #     print("ten_cards_indices")
    #
    #     #action_indices = np.zeros((14, 14, 14, 14, 14, 14, 14, 14, 14, 14), dtype=int)
    #
    #     inverse_indices = np.zeros((1716, 9), dtype=int)
    #
    #     i = 0
    #
    #     for c1 in range(7):
    #         #n1 = min(c1 + 4, 13)
    #         for c2 in range(c1 + 1, 13):
    #             for c3 in range(c2 + 1, 13):
    #                 for c4 in range(c3 + 1, 13):
    #                     for c5 in range(c4 + 1, 13):
    #                         for c6 in range(c5 + 1, 13):
    #                             for c7 in range(c6 + 1, 13):
    #                                 for c8 in range(c7 + 1, 13):
    #                                     for c9 in range(c8 + 1, 13):
    #                                         for c10 in range(c9 + 1, 13):
    #                         # print(f'c1: {c1}, c2: {c2}, c3: {c3}, c4: {c4}')
    #                                            # action_indices[c1, c2, c3, c4, c5, c6, c7, c8, c9] = i
    #                         # inverse_indices[i, :] = [c1, c2, c3, c4, c5]
    #                                             i += 1

    print(f'i: {i}')
    # print("action_indices:")
    # print(action_indices)
    print("\ninverse_indices:")
    print(inverse_indices)

def createIndices():

    indices = []

    indices.extend(two_ladder_indices())
    indices.extend(two_cards_phoenix_indices())
    indices.extend(three_cards_indices())
    indices.extend(three_cards_phoenix_indices())
    indices.extend(four_card_bomb())
    indices.extend(two_ladder_indices())
    indices.extend(two_ladder_phoenix_indices())
    indices.extend(full_house_indices())
    indices.extend(full_house_phoenix_indices())
    indices.extend(six_cards_indices())
    indices.extend(seven_cards_indices())
    indices.extend(eight_cards_indices())
    indices.extend(nine_cards_indices())

    return indices

if __name__ == "__main__":


    indices = createIndices()
    print(f'Total indices: {len(indices)}')

    # two_cards_indices()
    # two_cards_phoenix_indices()
    # three_cards_indices()
    # three_cards_phoenix_indices()
    # four_card_bomb()
    # two_ladder_indices()
    # two_ladder_phoenix_indices()
    # full_house_indices()
    # full_house_phoenix_indices()
    # six_cards_indices()  #
    # seven_cards_indices()
    # eight_cards_indices()
    # nine_cards_indices()
    # ten_cards_indices()
    # TODO: the system can't do more than 9 cards at the moment
