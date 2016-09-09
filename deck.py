import random


class Deck:
    def __init__(self):
        self.unused_deck = []
        self.used_deck = []
        self.reset_deck()

    def reset_deck(self):
        self.used_deck = []
        self.unused_deck = []
        for color in range(4):
            for value in range(2, 15):
                self.unused_deck.append([color, value])

    def draw_card(self):
        card_id = random.randint(0, len(self.unused_deck) - 1)
        card = self.unused_deck[card_id]
        self.unused_deck.remove(card)
        self.used_deck.append(card)
        return card

    def draw_specific_card(self, color, value):
        for c in self.unused_deck:
            if c[0] == color and c[1] == value:
                self.unused_deck.remove(c)
                self.used_deck.append(c)
                return c
        return None
