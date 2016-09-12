import random


class Deck:
    def __init__(self):
        self.unused_deck = []
        self.used_deck = []
        self.reset_deck()

    def reset_deck(self):
        self.used_deck = []
        self.unused_deck = [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12],
                            [0, 13], [0, 14], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10],
                            [1, 11], [1, 12], [1, 13], [1, 14], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
                            [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6],
                            [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14]]

    def draw_card(self):
        card_id = random.randint(0, len(self.unused_deck) - 1)
        card = self.unused_deck[card_id]
        self.unused_deck.remove(card)
        self.used_deck.append(card)
        return card

    def draw_n_cards(self, N):
        cards = []
        for i in range(N):
            cards.append(self.draw_card())
        return cards



    def draw_specific_card(self, color, value):
        for c in self.unused_deck:
            if c[0] == color and c[1] == value:
                self.unused_deck.remove(c)
                self.used_deck.append(c)
                return c
        return None
