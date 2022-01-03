# 1. pack: self.t x n dimensional vector such that the ith element is 1 if the ith card in the set is in the pack. You have a dictionary mapping these ids to card names.
# 2. shifted picks: self.t length vector where each element corresponds to the PRIOR pick. So the first element is always n + 1 (a card id that doesnt exist), and the second element is the card ID for what the human took P1P1
# 3. Positional information: literally just the list [0,1,2,...,41]
import numpy as np
import tensorflow as tf
import os
import pickle

MODEL_LOCATION = "Models/vow_emb_deep"

class RyanBot:
    def __init__(self):
        self.model = self.load_model_and_attrs(MODEL_LOCATION)
        self.clear_callback()
        # populate the predictions for every card in the set at p1p1 on init
        self.run_prediction(1, 1, full_set=True)
    def get_idx(self, pack, pick):
        idx = ((pack - 1) * self.pack_size) + pick - 1
        self.last_idx = idx
        return idx
    def pack_callback(self, pack, pick, pack_card_names):
        # print('pack_callback!', pack, pick, pack_card_names)
        print('pack_callback!', pack, pick)
        idx = self.get_idx(pack, pick)
        # run predictions without considering pack context
        self.run_prediction(pack, pick, full_set=True)
        # when we get to a pack we ensure all cards are empty
        # we do this because we initialize all packs to be every card
        self.pack[idx, :] = 0.0
        for name in pack_card_names:
            card_idx = self.get_card_idx(name)
            self.pack[idx, card_idx] = 1
        self.run_prediction(pack, pick)
    def pick_callback(self, pack, pick, card_name):
#        print('pick_callback!', pack, pick, card_name)
        print('pick_callback!', pack, pick)
        idx = self.get_idx(pack, pick)
        if idx + 1 < self.t:
            card_idx = self.get_card_idx(card_name)
            self.shifted_picks[idx + 1] = card_idx
        self.run_prediction(pack, pick)
    def clear_callback(self):
        print('clear_callback!')
        self.pack = np.ones((self.t, self.n_cards), dtype=np.float32)
        self.shifted_picks = np.ones(self.t, dtype=np.int32) * self.n_cards
        self.predictions = np.zeros((self.t, self.n_cards), dtype=np.float32)
        self.predictions_out_of_pack = np.zeros((self.t, self.n_cards), dtype=np.float32)
        self.positions = np.arange(self.t, dtype=np.int32)
        self.att_pack = None
        self.att_pick = None
        self.att_both = None
    def run_prediction(self, pack, pick, full_set=False):
        #print(f"running prediction for P{pack}P{pick} with full_set={full_set}")
        model_input = (
            np.expand_dims(self.pack, 0),
            np.expand_dims(self.shifted_picks, 0),
            np.expand_dims(self.positions, 0),
        )
        model_output, att = self.model(model_input, training=False, return_attention=True)
        idx = self.get_idx(pack, pick)
        prediction = np.squeeze(model_output)[idx]
        #self.rating(idx, self.idx_to_name[prediction])
        if full_set:
            self.predictions_out_of_pack[idx] = prediction
        else:
            self.predictions[idx] = prediction
            self.att_pack = att[0]
            self.att_pick = att[1][0]
            self.att_both = att[1][1]

    def get_att_vec(self, pack, pick, which=None,slice_leading_zeros=False):
        idx = self.get_idx(pack, pick)
        if which == "pack":
            att = self.att_pack
        elif which == "pick":
            att = self.att_pick
        else:
            att = self.att_both
        out = att[0,:,idx,:idx + 1]
        if slice_leading_zeros:
            start_idx = np.where(out >= 1e-4)[1]
            if len(start_idx) == 0:
                slice_idx = 0
            else:
                slice_idx = min(start_idx)
            out = (out, slice_idx)
        return out
    def rating(self, idx, card_name, full_set=False):
#        print(self.predictions)
        card_idx = self.get_card_idx(card_name)
        if full_set:
            card_rating = self.predictions_out_of_pack[idx, card_idx]
        else:
            card_rating = self.predictions[idx, card_idx]
        return card_rating
    def get_card_idx(self, card_name):
        name = card_name.lower().split("//")[0].strip()
        return self.name_to_idx[name]
    def load_model_and_attrs(self, location):
        model_loc = os.path.join(location, "model")
        data_loc = os.path.join(location, "attrs.pkl")
        model = tf.saved_model.load(model_loc)
        with open(data_loc, "rb") as f:
            attrs = pickle.load(f)
        self.t = attrs['t']
        self.pack_size = self.t//3
        self.n_cards = attrs['n_cards']
        self.idx_to_name = attrs['idx_to_name']
        self.name_to_idx = {v:k for k,v in self.idx_to_name.items()}
        return model
