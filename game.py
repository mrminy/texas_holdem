import time

from player import Player
import parameters
import evaluator
from deck import Deck


class Texas_holdem:
    def __init__(self, input_players=[], logger=False):
        self.deck = Deck()
        self.board = []
        self.pot = 0
        self.all_betting_history = []
        self.players = []
        self.players_this_round = []
        self.round_nr = 0
        self.current_player = None
        self.current_bet = 0
        self.deal_nr = 0
        self.previous_raise = 0
        self.all_in_nr = -1
        self.min_raise = 0
        self.big_blind_id = 0
        self.logger = logger
        self.reset(input_players)

    def reset(self, input_players=[]):
        self.deck.reset_deck()
        self.board = []
        self.all_betting_history = []
        self.pot = 0
        self.round_nr = 0
        self.deal_nr = 0
        self.all_in_nr = -1
        self.current_player = None
        self.current_bet = 0
        self.min_raise = 0
        self.previous_raise = 0
        self.big_blind_id = 0
        self.players_this_round = []
        self.players = []
        if len(input_players) <= 1:
            for i in range(parameters.NR_OF_PLAYERS):
                self.players.append(
                    Player(i, str(i), parameters.START_CHIPS))
        else:
            for p in input_players:
                p.chips = parameters.START_CHIPS
                self.players.append(p)
        self.players[0].blind = 1
        self.players[1].blind = 2
        self.new_round()

    def get_board_card_by_index(self, index):
        if len(self.board) > index:
            return self.board[index]
        return None

    def new_round(self):
        if self.logger:
            print(self)
            if self.players_this_round is not None and len(self.players_this_round) > 0:
                print("Winners of round:", self.players_this_round)
        if len(self.players) > 1:
            self.players_this_round = []
            self.deal_nr = 0
            self.round_nr += 1
            self.deck.reset_deck()
            self.board = []
            self.pot = 0
            self.previous_raise = 0
            self.all_in_nr = -1
            self.all_betting_history.append([])
            self.min_raise = parameters.MIN_BET

            # Switching blinds in group
            for p in self.players:
                if p.id_value == self.big_blind_id:
                    p.blind = 1
                    next_p = self.get_next_player(p, self.players)
                    next_p.blind = 2
                    self.big_blind_id = next_p.id_value
                    if len(self.players_this_round) > 2:
                        self.get_previous_player(p, self.players).blind = 0
                    break
            # Making ready for new round (paying blinds)
            for i, p in enumerate(self.players):
                self.players_this_round.append(p)
                blind_value = 0
                p.bet = 0
                if p.blind == 2:
                    blind_value = min(parameters.BIG_BLIND, self.find_max_raise())
                elif p.blind == 1:
                    blind_value = min(parameters.SMALL_BLIND, self.find_max_raise())
                self.pot += blind_value
                p.bet += blind_value
                p.chips -= blind_value
                p.give_hand(self.deck.draw_card(), self.deck.draw_card())
            self.current_bet = parameters.BIG_BLIND

    def dealer_step(self):
        if self.deal_nr == 0:
            # Deal 3 cards
            self.board.append(self.deck.draw_card())
            self.board.append(self.deck.draw_card())
            self.board.append(self.deck.draw_card())
        if self.deal_nr == 1 or self.deal_nr == 2:
            # Deal 1 card
            self.board.append(self.deck.draw_card())
        for p in self.players:
            p.bet = 0
        self.deal_nr += 1

    def who_wins(self):
        if len(self.players_this_round) == 0 and self.logger:
            print("ERROR!")
            return None
        elif len(self.players_this_round) == 1:
            self.deal_pot([self.players_this_round[0]])
            return self.players_this_round[0]
        player_score_map = []
        for player in self.players_this_round:
            player_score_map.append([evaluator.evaluate(self.fetch_cards_for_player(player)), player])
        player_score_map.sort(key=lambda x: x[0], reverse=True)
        top_score = player_score_map[0][0]
        top_players = [player_score_map[0][1]]
        counter = 1
        while counter < len(self.players_this_round) and player_score_map[counter][0] == top_score:
            top_players.append(player_score_map[counter][1])
            counter += 1
        self.deal_pot(top_players)
        if self.logger:
            print("Winners of round:", top_players)
        return top_players

    def find_players_from_id(self, id_values):
        players_from_id = []
        for p in self.players:
            for value in id_values:
                if p.id_value == value:
                    players_from_id.append(p)
        return players_from_id

    def find_players_from_id_in_list(self, id_value, list_of_players):
        for p in list_of_players:
            if p.id_value == id_value:
                return p
        return None

    def find_player_from_id(self, id_value):
        for p in self.players:
            if p.id_value == id_value:
                return p
        return None

    def clear_players(self):
        for p in self.players:
            if p.chips <= 0:
                if p.blind == 2:
                    self.get_next_player(p, self.players).blind = 2
                self.players.remove(p)

    def deal_pot(self, top_players):
        # One of these players wins
        if len(top_players) == 1:
            top_players[0].chips += self.pot
            self.pot = 0
        elif len(top_players) != 0:
            split = int(self.pot / len(top_players))
            for p in top_players:
                p.chips += split
            self.pot = 0
            # print("whut")

    def fetch_cards_for_player(self, player):
        cards = []
        for card in self.board:
            cards.append(card)
        cards.append(player.hand[0])
        cards.append(player.hand[1])
        return cards

    def get_player_after_big_blind(self):
        for i, p in enumerate(self.players_this_round):
            if p.blind == 2:
                return self.get_next_player(p, self.players_this_round)
        return self.players_this_round[0]

    def get_next_player(self, current_player, player_array):
        if current_player is None:
            return player_array[0]
        index = current_player.id_value
        for i, p in enumerate(player_array):
            if p.id_value > index:
                return p
        return player_array[0]

    def reverse_enum(self, L):
        for index in reversed(range(len(L))):
            yield index, L[index]

    def get_previous_player(self, current_player, player_array):
        index = current_player.id_value
        for i, p in self.reverse_enum(player_array):
            if p.id_value < index:
                return p
        return player_array[len(player_array) - 1]

    def all_pleased(self):
        if len(self.players_this_round) == 1:
            return True
        if self.is_all_in() and self.deal_nr != self.all_in_nr:
            return True
        for p in self.players_this_round:
            if p.bet < self.current_bet:
                return False
        return True

    def find_min_bet(self):
        min_value = parameters.MIN_BET
        for p in self.players_this_round:
            if p.chips < parameters.MIN_BET:
                min_value = p.chips
        return min_value

    def find_max_bet(self):
        max_value = 999999999
        for p in self.players_this_round:
            if p.bet + p.chips < max_value:
                max_value = p.chips + p.bet
        return max_value

    def find_max_raise(self):
        max_raise = 999999999
        for p in self.players_this_round:
            if p.chips < max_raise:
                max_raise = p.chips
        return max_raise

    def is_all_in(self):
        for p in self.players_this_round:
            if p.chips <= 0:
                return True
        return False

    def bet(self):
        board_copy = self.board[:]
        open_information_players_this_round = []
        max_bet = self.find_max_bet()
        min_bet = self.find_min_bet()
        for p_this_round in self.players_this_round:
            open_information_players_this_round.append(p_this_round.get_open_information())
        start_time = time.time()
        original_bet = self.current_player.make_decision(self.all_betting_history[-1], int(self.current_bet),
                                                         int(max_bet), int(min_bet), int(0),
                                                         open_information_players_this_round, int(self.pot), board_copy)
        used_time = time.time()-start_time
        if used_time > 1.0:
            print("Player used longer than 1 second to decide. Counts as fold...", self.current_player)
            original_bet = -1
        if original_bet is None:
            print(self.current_player, "ERROR!! Returned None!!")
            original_bet = -1
        original_bet = int(original_bet)
        if self.logger:
            print(self.current_player, "original bet:", original_bet)

        modded_bet = min(min(original_bet, self.current_player.chips), self.find_max_bet())
        is_all_in = self.is_all_in()
        if is_all_in and modded_bet + self.current_player.bet > self.current_bet:
            modded_bet = self.current_bet - self.current_player.bet

        if modded_bet + self.current_player.bet > self.current_bet:
            # Raising
            if modded_bet < self.previous_raise:
                if self.logger:
                    print("bet is lower than previous raise", original_bet, modded_bet, self.previous_raise)
                modded_bet = self.previous_raise
            if modded_bet < min_bet:
                # Raising too low
                if self.logger:
                    print("bet is lower than min_bet", original_bet, modded_bet, min_bet)
                modded_bet = min_bet
            elif modded_bet > max_bet:
                # Raising too high
                if self.logger:
                    print("bet is higher than max_bet", original_bet, modded_bet, self.current_player.bet, max_bet)
                modded_bet = min(self.current_player.chips, max_bet)
            self.current_player.bet += modded_bet
            self.current_player.chips -= modded_bet
            self.pot += modded_bet
            self.previous_raise = modded_bet
            self.current_bet = self.current_player.bet
            if self.logger:
                print(self.current_player, "raising...", modded_bet)
        elif modded_bet + self.current_player.bet == self.current_bet:
            # Calling
            if modded_bet < min_bet and modded_bet != 0:
                modded_bet = min_bet
            self.current_player.bet += modded_bet
            self.current_player.chips -= modded_bet
            self.pot += modded_bet
            if self.logger:
                print(self.current_player, "calling...", modded_bet)
        elif modded_bet == 0 and self.current_bet == 0:
            # Checking
            if self.logger:
                print(self.current_player, "checking...")
            pass
        else:
            if self.current_bet != 0:
                # Folding
                if self.logger:
                    print(self.current_player, "folding...")
                self.current_player.bet = 0
                self.players_this_round.remove(self.current_player)
        if self.current_player.chips <= 0:
            self.all_in_nr = self.deal_nr
        self.all_betting_history[-1].append([modded_bet, self.current_player.id_value])

    def play_one_step(self):
        if len(self.players) == 1:
            if self.logger:
                print("Finished! Only one left.", self.players[0])
            return
        if self.deal_nr >= 4 or len(self.players_this_round) == 1:
            if self.logger:
                print("Pot: ", self.pot, " - board: ", self.board)
            self.who_wins()
            self.clear_players()
            if len(self.players) == 1:
                print("We have a winner!", self.players[0], "Rounds played: " + str(self.round_nr))
                return
            if self.logger:
                print("New round, after full flop...")
            self.new_round()
            return
        if self.all_pleased():
            # Deal cards...
            if self.logger:
                print("All pleased. Deal cards")
            self.dealer_step()
            return

        # Make next player bid, fold or check
        self.current_player = self.get_next_player(self.current_player, self.players_this_round)
        if self.current_player.chips != 0:
            if self.logger:
                print("Betting...", self.current_player)
            self.bet()
            if self.logger:
                print(self)

    def __str__(self):
        str_out = "Round nr: " + str(self.round_nr) + ", deal nr: " + str(self.deal_nr) + ", board: " + str(self.board)
        if len(self.players[0].hand) > 0:
            for p in self.players:
                str_out += " - " + str(p)
        return str_out
