import parameters


def evaluate(cards):
    if len(cards) < 5:
        return None
    value, top_cards = check_straight_flush(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = check_four_of_a_kind(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = check_full_house(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = check_flush(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = check_straight(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = check_three_of_a_kind(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = check_two_pairs(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = check_one_pair(cards)
    if value != 0:
        return generate_score(value, top_cards)
    value, top_cards = high_card(cards)
    return generate_score(value, top_cards)


def generate_score(value, top_cards):
    score_arr = [value]
    for card in top_cards:
        score_arr.append(card[1])
    return score_arr


def evaluate_score(score_arr_1, score_arr_2):
    for i in range(len(score_arr_1)):
        if score_arr_1[i] > score_arr_2[i]:
            return 1
        elif score_arr_1[i] < score_arr_2[i]:
            return -1
    return 0


def check_straight_flush(cards):
    cards.sort(key=lambda x: x[0], reverse=True)
    for i in range(0, len(cards) - 4):
        if cards[i][0] == cards[i + 1][0] == cards[i + 2][0] == cards[i + 3][0] == cards[i + 4][0]:
            flush_cards = cards[i:i + 5]
            value, straight_cards = check_straight(flush_cards)
            if value != 0:
                return parameters.EVAL_VALUES[0], straight_cards
    return 0, []


def check_four_of_a_kind(cards):
    cards.sort(key=lambda x: x[1], reverse=True)
    top_cards = []
    for i in range(0, len(cards) - 3):
        if cards[i][1] == cards[i + 1][1] == cards[i + 2][1] == cards[i + 3][1]:
            top_cards.append(cards[i])
            top_cards.append(cards[i + 1])
            top_cards.append(cards[i + 2])
            top_cards.append(cards[i + 3])
            if i == 0:
                top_cards.append(cards[i + 4])
            else:
                top_cards.append(cards[0])
            return parameters.EVAL_VALUES[1], top_cards
    return 0, []


def check_full_house(cards):
    cards.sort(key=lambda x: x[1], reverse=True)
    full_house_cards = []
    ids = []
    for i in range(0, len(cards) - 2):
        if cards[i][1] == cards[i + 1][1] == cards[i + 2][1]:
            full_house_cards.append(cards[i])
            full_house_cards.append(cards[i + 1])
            full_house_cards.append(cards[i + 2])
            ids = [i, i + 1, i + 2]
            break
    if len(ids) == 3:
        for j in range(0, len(cards) - 1):
            if cards[j][1] == cards[j + 1][1] and j not in ids and j + 1 not in ids:
                full_house_cards.append(cards[j])
                full_house_cards.append(cards[j + 1])
                # full_house_cards.sort(key=lambda x: x[1], reverse=True) # What is highest in full house?
                return parameters.EVAL_VALUES[2], full_house_cards
    return 0, []


def check_flush(cards):
    cards.sort(key=lambda x: x[0], reverse=True)
    for i in range(0, len(cards) - 4):
        if cards[i][0] == cards[i + 1][0] == cards[i + 2][0] == cards[i + 3][0] == cards[i + 4][0]:
            flush_cards = cards[i:i + 5]
            flush_cards.sort(key=lambda x: x[1], reverse=True)
            return parameters.EVAL_VALUES[3], flush_cards
    return 0, []


def check_straight(cards):
    cards.sort(key=lambda x: x[1], reverse=True)
    for i, c in enumerate(cards):
        counter = 0
        straight_cards = [c]
        current_value = c[1]
        while counter < 4:
            for j in range(i + 1, len(cards)):
                if cards[j][1] + 1 == current_value:
                    current_value = cards[j][1]
                    straight_cards.append(cards[j])
            counter += 1
        if len(straight_cards) == 5:
            return parameters.EVAL_VALUES[4], straight_cards
    # Handle ace...
    if cards[0][1] == 14:
        for i, c in enumerate(cards):
            counter = 0
            straight_cards = [c]
            current_value = c[1]
            while counter < 3:
                for j in range(i + 1, len(cards)):
                    if cards[j][1] + 1 == current_value:
                        current_value = cards[j][1]
                        straight_cards.append(cards[j])
                counter += 1
            if len(straight_cards) == 4:
                straight_cards.append(cards[0])
                return parameters.EVAL_VALUES[4], straight_cards
    return 0, []


def check_three_of_a_kind(cards):
    cards.sort(key=lambda x: x[1], reverse=True)
    top_cards = []
    for i in range(0, len(cards) - 2):
        i2, i3 = i + 1, i + 2
        if cards[i][1] == cards[i2][1] == cards[i3][1]:
            top_cards.append(cards[i])
            top_cards.append(cards[i + 1])
            top_cards.append(cards[i + 2])
            for j in range(0, len(cards)):
                if j != i and j != i2 and j != i3 and len(top_cards) < 5:
                    top_cards.append(cards[j])
            top_cards[3:len(top_cards)] = sorted(top_cards[3:len(top_cards)], key=lambda x: x[1], reverse=True)
            return parameters.EVAL_VALUES[5], top_cards
    return 0, []


def check_two_pairs(cards):
    cards.sort(key=lambda x: x[1], reverse=True)
    pairs = []
    ids = []
    for i, c1 in enumerate(cards):
        if i not in ids:
            for j, c2 in enumerate(cards):
                if i != j and c1[1] == c2[1] and c1[0] != c2[0] and j not in ids and i not in ids:
                    pairs.append(c1)
                    pairs.append(c2)
                    ids.append(i)
                    ids.append(j)
                if len(pairs) == 4:
                    for k in range(0, len(cards)):
                        if len(pairs) >= 5:
                            break
                        elif k not in ids and len(pairs) < 5:
                            pairs.append(cards[k])
                    pairs[4:len(pairs)] = sorted(pairs[4:len(pairs)], key=lambda x: x[1], reverse=True)
                    return parameters.EVAL_VALUES[6], pairs
    return 0, []


def check_one_pair(cards):
    cards.sort(key=lambda x: x[1], reverse=True)
    pairs = []
    ids = [-1, -1]
    for i in range(0, len(cards) - 1):
        if cards[i][1] == cards[i + 1][1]:
            ids = [i, i + 1]
            pairs.append(cards[i])
            pairs.append(cards[i + 1])
        if len(pairs) == 2:
            for k in range(0, len(cards)):
                if k != ids[0] and k != ids[1] and len(pairs) < 5:
                    pairs.append(cards[k])
            pairs[2:len(pairs)] = sorted(pairs[2:len(pairs)], key=lambda x: x[1], reverse=True)
            return parameters.EVAL_VALUES[7], pairs
    return 0, []


def high_card(cards):
    cards.sort(key=lambda x: x[1], reverse=True)
    return parameters.EVAL_VALUES[8], cards[0:5]

