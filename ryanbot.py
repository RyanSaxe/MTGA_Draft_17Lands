# 1. pack: 42 x n dimensional vector such that the ith element is 1 if the ith card in the set is in the pack. You have a dictionary mapping these ids to card names.
# 2. shifted picks: 42 length vector where each element corresponds to the PRIOR pick. So the first element is always n + 1 (a card id that doesnt exist), and the second element is the card ID for what the human took P1P1
# 3. Positional information: literally just the list [0,1,2,...,41]
import numpy as np

N_CARDS = 256

def lookup_card(card_name):
    # replace w/ actual card name to number function
    return hash(card_name) % N_CARDS

def predict(pack_number, pick_number, pack, shifted_picks):
    # this should run the model, returning "a vector of length n that sums to 1 where each value corresopnds to the score of that card"
    idx = ((pack_number - 1) * 14) + pick_number
    return pack[idx].copy()

class RyanBot:
    def __init__(self):
        self.clear_callback()
    def pack_callback(self, pack, pick, pack_card_names):
        print('pack_callback!', pack, pick, pack_card_names)
        idx = ((pack - 1) * 14) + pick - 1
        print(idx)
        current_pack = self.pack[idx]
        for name in pack_card_names:
            current_pack[lookup_card(name)] = 1
        self.run_prediction(pack, pick)
    def pick_callback(self, pack, pick, card_name):
        print('pick_callback!', pack, pick, card_name)
        idx = ((pack - 1) * 14) + pick
        print(idx)
        current_picks = self.shifted_picks[idx]
        current_picks[lookup_card(card_name)] = 1
    def clear_callback(self):
        print('clear_callback!')
        self.pack = []
        self.shifted_picks = []
        self.predictions = []
        for i in range(0, 42):
            self.pack.append(np.zeros(N_CARDS))
            self.shifted_picks.append(np.zeros(N_CARDS))
            self.predictions.append(np.zeros(N_CARDS))
    def run_prediction(self, pack, pick):
        idx = ((pack - 1) * 14) + pick
        result = predict(pack, pick, self.pack, self.shifted_picks)
        self.predictions[idx] = result.copy()
    def rating(self, pick_number, card_name):
        print('rating', pick_number, card_name)
#        print(self.predictions)
        return 42
#        return self.predictions[pick_number][lookup_card(card_name)]
        
