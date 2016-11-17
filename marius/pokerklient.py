"""
Coded by Marius Amundsen
"""

import random



class deck(object):
    """ One hole deck of playing cards"""

    def __init__(self):
        self.deck = [
            [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14],
            # Spade
            [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14],
            # heart
            [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14],
            # diomonds
            [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13],
            [3, 14]]  # club

    def card(self, id):
        return self.deck[id][0], self.deck[id][1]

    def cards(self, id):
        result = ()
        for a in id:
            result += self.deck[a][0], self.deck[a][1]
        return result

    def find_card_id(self, cards):
        card_index = []
        for card in cards:
            for index in range(len(self.deck)):
                if card == self.deck[index]:
                    card_index.append(index)

        return card_index


class winner(object):
    """Holding status over best find cards combansioin"""

    def __init__(self, players):
        self.players = players
        self.playerScore = [-111111111111111, -111111111111111,-111111111111111,-111111111111111,-111111111111111]

    def update_Score(self, player, score):
        if self.playerScore[player] < score: self.playerScore[player] = score

    def see_Score(self, player):
        return self.playerScore[player]

    def find_Winner(self):
        player = 11
        score = -111111111111111
        for a in range(self.players):
            if self.playerScore[a] > score and self.playerScore[a] != 0:
                score = self.playerScore[a]
                player = a
            elif self.playerScore[a] == score > 0:
                player = 12

        return player, score





class rules(object):
    """ Finding a Winner"""

    def __init__(self, cardsin, players):
        findcard = deck()
        self.winner = winner(players)
        for a in range(players):

            counterV = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            counterC = [0, 0, 0, 0]

            for b in range(7):
                if len(cardsin) > a * 7 + b:
                    update = findcard.card(cardsin[a * 7 + b])
                    counterV[update[1] - 2] += 1
                    counterC[update[0]] += 1

                    # Straight flush
            straightflushCounter = 1
            for c in counterV:
                if c > 0:
                    straightflushCounter += 1
                else:
                    straightflushCounter = 1
                if straightflushCounter > 5:
                    for d in counterC:
                        if d > 4:
                            straightflush = sorted(cardsin[a * 7:a * 7 + 7])
                            last = 0
                            counter = 0
                            for e in straightflush:
                                if e == last + 1:
                                    last = e
                                    counter += 1
                                else:
                                    last = e
                                    counter = 0
                                if counter > 4:
                                    self.winner.update_Score(a, 1000000000 + findcard.card(e)[1])

                                    counter = 0

                                    # Four of a kind

            for index, counterT in enumerate(counterV, start=0):
                if counterT == 4:
                    self.winner.update_Score(a, 100000000 + index * 1000)

                    # Full house
            for index1, counterT in enumerate(counterV, start=0):
                if counterT == 3:
                    for index2, counterT in enumerate(counterV, start=0):
                        if counterT == 2:
                            self.winner.update_Score(a, 10000000 + 1000 * (index1 * 10) + index2)

                            # Flush

            for index3, counterT in enumerate(counterC, start=0):
                if counterT > 4:
                    flush = []
                    for g in range(7):
                        if len(cardsin) > a * 7 + g:
                            update = findcard.card(cardsin[a * 7 + g])
                            if update[0] == index3:
                                flush.append(update[1])
                                flush.sort()
                    self.winner.update_Score(a, 1000000 + flush[4] * 10000 + flush[3] * 1000 + flush[2] * 100 + flush[
                        1] * 10 + flush[0])

                    # Straight

            if counterV[12] > 0:
                straightCounter = 1  # Check if there are an ace
            else:
                straightCounter = 0
            for index4, counterT in enumerate(counterV, start=0):
                if counterT > 0:
                    straightCounter += 1
                else:
                    straightCounter = 0
                if straightCounter > 4:
                    self.winner.update_Score(a, 100000 + index * 1000)
                    # Three of a kind
            for index1, counterT in enumerate(counterV, start=0):
                if counterT == 3:
                    self.winner.update_Score(a, 10000 + index1 * 1000)
                    # Two pair
                    # Sender her ut riktig verdi,start =2
            twopair = [0, 0]
            for index1, counterT in enumerate(counterV, start=0):
                if counterT == 2:
                    twopair[0] = twopair[1]
                    twopair[1] = index1

            if twopair[0] > 0:
                self.winner.update_Score(a, 1000 + 10 * (twopair[1] * 10) + twopair[0])
            else:
                # Pair
                onepair = 0
                for index1, counterT in enumerate(counterV, start=0):
                    if counterT == 2:
                        onepair = index1

                if onepair > 0:
                    self.winner.update_Score(a, 100 + index1)
                    # High card

            highcard = [0, 0, 0, 0, 0]
            for index1, counterT in enumerate(counterV, start=0):
                if counterT == 1:
                    highcard[0] = highcard[1]
                    highcard[1] = highcard[2]
                    highcard[2] = highcard[3]
                    highcard[3] = highcard[4]

                    highcard[4] = index1
            if highcard[0] > 0:
                self.winner.update_Score(a, (
                    (1 / highcard[4]) * 100000 + (1 / highcard[3]) * 10000 + (1 / highcard[2]) * 1000 + (
                        1 / highcard[2]) * 100 + (1 / highcard[1]) * 10) * -1)

            # Highcard Needed
            ekstraCardsScore = self.winner.see_Score(a)
            ekstraCard = 0
            if 10000 < ekstraCardsScore < 100000:
                self.winner.update_Score(a, ekstraCardsScore + (highcard[4]) * 10 + (highcard[3]))
            elif 1000 < ekstraCardsScore < 10000:
                self.winner.update_Score(a, ekstraCardsScore + highcard[4])
            elif 100 < ekstraCardsScore < 1000:
                self.winner.update_Score(a, ekstraCardsScore + (highcard[4] * 10) + (highcard[3] * 5) + (highcard[2]))

    def see_winner(self):
        return self.winner.find_Winner()

