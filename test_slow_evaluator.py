import unittest
import evaluator


class TestEvaluator(unittest.TestCase):
    def test_high_card(self):
        # Test high card
        cards_1 = [[0, 2], [1, 3], [1, 5], [2, 8], [3, 10]]
        cards_2 = [[0, 3], [1, 6], [1, 5], [2, 8], [3, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("High card:", cards_1_score, cards_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(cards_1_score, cards_2_score))

    def test_one_pair(self):
        # Test 1 pair
        pair_1_1 = [[0, 2], [1, 2], [1, 5], [2, 8], [3, 10]]
        pair_1_2 = [[0, 9], [1, 9], [1, 5], [2, 8], [3, 10]]
        pair_1_1_score = evaluator.evaluate(pair_1_1)
        pair_1_2_score = evaluator.evaluate(pair_1_2)
        print("One pair:", pair_1_1_score, pair_1_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(pair_1_1_score, pair_1_2_score))

        pair_1_1 = [[0, 2], [1, 2], [1, 5], [2, 8], [3, 10], [0, 7]]
        pair_1_2 = [[0, 2], [1, 2], [1, 5], [2, 8], [3, 10], [0, 6]]
        pair_1_1_score = evaluator.evaluate(pair_1_1)
        pair_1_2_score = evaluator.evaluate(pair_1_2)
        print("One pair reverse:", pair_1_1_score, pair_1_2_score)
        self.assertEqual(1, evaluator.evaluate_score(pair_1_1_score, pair_1_2_score))

        pair_1_1 = [[0, 2], [1, 2], [1, 5], [2, 8], [3, 10], [0, 7]]
        pair_1_2 = [[0, 2], [1, 2], [1, 4], [2, 8], [3, 10], [0, 7]]
        pair_1_1_score = evaluator.evaluate(pair_1_1)
        pair_1_2_score = evaluator.evaluate(pair_1_2)
        print("One pair equal:", pair_1_1_score, pair_1_2_score)
        self.assertEqual(0, evaluator.evaluate_score(pair_1_1_score, pair_1_2_score))

    def test_two_pair(self):
        # Test 2 pair
        pair_2_1 = [[0, 2], [1, 2], [1, 8], [2, 8], [3, 10]]
        pair_2_2 = [[0, 9], [1, 9], [1, 5], [2, 5], [3, 10]]
        pair_2_1_score = evaluator.evaluate(pair_2_1)
        pair_2_2_score = evaluator.evaluate(pair_2_2)
        print("Two pair:", pair_2_1_score, pair_2_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(pair_2_1_score, pair_2_2_score))

        pair_2_1 = [[0, 2], [1, 2], [1, 9], [2, 9], [3, 10], [1, 10], [2, 14]]
        pair_2_2 = [[0, 9], [1, 9], [1, 5], [2, 5], [3, 10], [1, 10], [2, 14]]
        pair_2_1_score = evaluator.evaluate(pair_2_1)
        pair_2_2_score = evaluator.evaluate(pair_2_2)
        print("Two pair equal:", pair_2_1_score, pair_2_2_score)
        self.assertEqual(0, evaluator.evaluate_score(pair_2_1_score, pair_2_2_score))

    def test_three_of_a_kind(self):
        # Test 3 of a kind
        cards_1 = [[0, 2], [1, 2], [3, 2], [2, 8], [3, 10]]
        cards_2 = [[0, 9], [3, 5], [1, 5], [2, 5], [3, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Three of a kind:", cards_1_score, cards_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(cards_1_score, cards_2_score))

    def test_straight(self):
        # Test straight
        cards_1 = [[0, 1], [1, 2], [3, 3], [2, 4], [3, 5]]
        cards_2 = [[0, 13], [3, 12], [1, 11], [2, 9], [3, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Straight:", cards_1_score, cards_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(cards_1_score, cards_2_score))

    def test_flush(self):
        # Test flush
        cards_1 = [[0, 2], [0, 3], [0, 5], [0, 8], [0, 10]]
        cards_2 = [[3, 5], [3, 2], [3, 11], [3, 9], [3, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Flush:", cards_1_score, cards_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(cards_1_score, cards_2_score))

        cards_1 = [[0, 2], [0, 3], [0, 5], [0, 8], [0, 10], [0, 12]]
        cards_2 = [[3, 5], [3, 2], [3, 11], [3, 9], [3, 10], [3, 4]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Flush reverse:", cards_1_score, cards_2_score)
        self.assertEqual(1, evaluator.evaluate_score(cards_1_score, cards_2_score))

    def test_four_of_a_kind(self):
        # Test four of a kind
        cards_1 = [[0, 2], [1, 2], [3, 2], [2, 2], [3, 14]]
        cards_2 = [[0, 13], [0, 10], [1, 10], [2, 10], [3, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Four of a kind:", cards_1_score, cards_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(cards_1_score, cards_2_score))

        cards_1 = [[0, 2], [1, 2], [3, 2], [2, 2], [2, 14], [3, 14]]
        cards_2 = [[0, 2], [1, 2], [3, 2], [2, 2], [2, 14], [3, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Four of a kind equal:", cards_1_score, cards_2_score)
        self.assertEqual(0, evaluator.evaluate_score(cards_1_score, cards_2_score))

    def test_straight_flush(self):
        # Test four of a kind
        cards_1 = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
        cards_2 = [[3, 14], [3, 13], [3, 12], [3, 11], [3, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Straight flush:", cards_1_score, cards_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(cards_1_score, cards_2_score))

        cards_1 = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 10]]
        cards_2 = [[0, 1], [1, 2], [0, 3], [0, 4], [0, 6], [0, 10]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Straight flush error test:", cards_1_score, cards_2_score)
        self.assertEqual(1, evaluator.evaluate_score(cards_1_score, cards_2_score))

        cards_1 = [[0, 1], [1, 9], [0, 8], [0, 6], [0, 7], [0, 10]]
        cards_2 = [[0, 1], [1, 2], [0, 3], [0, 4], [0, 5], [0, 2]]
        cards_1_score = evaluator.evaluate(cards_1)
        cards_2_score = evaluator.evaluate(cards_2)
        print("Straight flush error test 2:", cards_1_score, cards_2_score)
        self.assertEqual(-1, evaluator.evaluate_score(cards_1_score, cards_2_score))
