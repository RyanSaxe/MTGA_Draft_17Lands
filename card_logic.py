import functools
import json
from itertools import combinations

def CompareRatings(a, b):
    try:
        creature_types = ["Creature", "Planeswalker"]
        a_creature = any(x in a["card"]["types"] for x in creature_types)
        b_creature = any(x in b["card"]["types"] for x in creature_types)
        
        #Always pick the bombs
        if((a["base_rating"] >= 4.5) or (b["base_rating"] >= 4.5)):
            return b["base_rating"] - a["base_rating"]
        elif(a["rating"] == b["rating"]):
            # Breaking ties with good any color > bad on-color > creature > lower cmc
            if((a["base_rating"] >= 4.0) and (b["base_rating"] <= 1.5)):
                return -1
            elif((b["base_rating"] >= 4.0) and (a["base_rating"] <= 1.5)):
                return 1
            else:
                if(a["color_bonus"] == b["color_bonus"]):
                    if(a_creature == b_creature):
                        return a["card"]["cmc"] - b["card"]["cmc"]
                    else:
                        return b_creature - a_creature
                else:
                    return b["color_bonus"] - a["color_bonus"]
        else:
            return b["rating"] - a["rating"]
    except Exception as error:
        print("CompareRatings Error: %s" % error)
    return 0

def ColorAffinity(colors, card):
    rating = card["rating"]
    if(rating  >= 1.5):
        for color in card["colors"]:
            if (color not in colors):
                colors[color] = 0
            colors[color] += rating
        
    return colors 
  
def ColorBonus (deck, deck_colors, card):
    #print("deck_colors: %s" % str(deck_colors))
    color_bonus_levels = [0.0, 0.2, 0.4, 0.6, 0.8,
                          1.0, 1.0, 2.0, 2.0, 2.0, 
                          2.0, 3.0, 3.0, 3.0, 3.0, 
                          4.0, 4.0, 4.0, 4.0, 4.0,
                          5.0, 5.0, 5.0, 6.0, 6.0];

    color_bonus_factor = 0.0
    
    color_bonus_level = 0.0
    #print("ColorBonus Start")
    try:                        
        card_colors = card["colors"]
        if(len(card_colors) == 0):
            color_bonus_factor = 0.5
        else:
            matching_colors = list(filter((lambda x : x in deck_colors[-1]), card_colors))
            color_bonus_factor = len(matching_colors) / len(card_colors)
            #print("card_colors: %s, matching_colors: %s, color_bonus: %s" % (str(card_colors), str(matching_colors), str(color_bonus_factor)))
        color_bonus_level = color_bonus_levels[min(len(deck), len(color_bonus_levels) - 1)]
        
    except Exception as error:
        print(error)
    
    return color_bonus_factor * color_bonus_level
    
def CurveBonus(deck, deck_colors, card, pick_number):
    curve_bonus_levels = [0.4, 0.4, 0.4, 0.4, 0.4,
                          0.6, 0.6, 0.6, 0.6, 0.6,
                          1.0, 1.0, 1.0, 1.5, 1.5,
                          2.0, 2.0, 3.0, 3.0, 3.0]

    curve_bonus = 0.0
    curve_bonus_factor = 0.0
    cmc_average = 3.04
    minimum_creature_count = 13
    minimum_noncreature_count = 6
    minimum_distribution = [0, 0, 4, 3, 2, 1, 0]
    
    try:
        matching_colors = list(filter((lambda x : x in deck_colors[0]), card["colors"]))
        
        if len(matching_colors):
            if any(x in card["types"] for x in ["Creature", "Planeswalker"]):
                card_colors_sorted = DeckColorSearch(deck, deck_colors[0], ["Creature", "Planeswalker"], True)
                card_colors_sorted = sorted(card_colors_sorted, key = lambda k: k["deck_colors"]["All Decks"]["gihwr"], reverse = True)
                
                cmc_total, count, distribution = ColorCmc(card_colors_sorted)
                curve_bonus = curve_bonus_levels[int(min(pick_number, len(curve_bonus_levels) - 1))]
                
                curve_bonus_factor = 1
                if(count > minimum_creature_count):
                    replaceable = [x for x in card_colors_sorted if (card["cmc"] <= x["cmc"] and (card["deck_colors"]["All Decks"]["gihwr"]> x["deck_colors"]["All Decks"]["gihwr"]))]
                    curve_bonus_factor = 0
                    if len(replaceable):
                        index = int(min(card["cmc"], len(distribution) - 1))
                        
                        if(distribution[index] < minimum_distribution[index]):
                            curve_bonus_factor = 1
                        else:
                            curve_bonus_factor = 0.5
    except Exception as error:
        print("CurveBonus Error: %s" % error)
        
    return curve_bonus * curve_bonus_factor
    
def DeckColorSearch(deck, search_colors, card_types, include_types, include_colorless, include_partial):
    card_color_sorted = {}
    main_color = ""
    combined_cards = []
    try:
        for card in deck:
            card_colors = CardColors(card["mana_cost"])
            
            if not card_colors:
                card_colors = card["colors"]
            
            if include_partial and len(card_colors):
                card_colors = card_colors[0] #Just use the first color
            
            if bool(card_colors) and (set(card_colors) <= set(search_colors)):
                main_color = card_colors[0]
              
                if((include_types and any(x in card["types"][0] for x in card_types)) or
                (not include_types and not any(x in card["types"][0] for x in card_types))):
                    
                    if main_color not in card_color_sorted.keys():
                        card_color_sorted[main_color] = []
                        
                    card_color_sorted[main_color].append(card)
                    
            if (bool(card_colors) == False) and include_colorless:
            
                if((include_types and any(x in card["types"][0] for x in card_types)) or
                (not include_types and not any(x in card["types"][0] for x in card_types))):
                    
                    combined_cards.append(card)
        
        for color in card_color_sorted:
            combined_cards.extend(card_color_sorted[color])
            
    except Exception as error:
        print("DeckColorSearch Error: %s" % error)
        #print(card)
    return combined_cards
    
def ColorCmc(deck):
    cmc_total = 0
    count = 0
    distribution = [0, 0, 0, 0, 0, 0, 0]
    
    try:
        for card in deck:
            cmc_total += card["cmc"]
            count += 1
            index = int(min(card["cmc"], len(distribution) - 1))
            distribution[index] += 1
    
    except Exception as error:
        print("ColorCmc Error: %s" % error)
    
    return cmc_total, count, distribution
    
def ColorFilter(deck, color_selection, color_list):
    color_data = color_selection
    filtered_color = color_data
    try:
        if color_data == "Auto":
            filtered_color = AutoColors(deck, color_list, 2)
        elif color_data == "AI":
            filtered_color = "AI"
        elif color_data == "RyanBot":
            filtered_color = "RyanBot"
        else:
            filtered_color = [key for key, value in color_list.items() if value == color_data][0]
    except Exception as error:
        print("ColorFilter Error: %s" % error)
    return filtered_color
    
def DeckColors(deck, color_options, colors_max):
    try:
        deck_colors = []
        
        colors = CalculateColorAffinity(deck,"All Decks",50)
        
        # Modify the dictionary to include ratings
        colors = list(map((lambda x : {"color" : x, "rating" : colors[x]}), colors.keys()))
        
        # Sort the list by decreasing ratings
        colors = sorted(colors, key = lambda k : k["rating"], reverse = True)
        
        # Remove extra colors beyond limit
        colors = colors[0: colors_max]
        
        # Return colors 
        sorted_colors = list(map((lambda x : x["color"]), colors))
        
        #sorted_colors)
        
        #Create color permutation
        color_combination = []
        
        for count in range(colors_max + 1):
            if count > 1:
                color_combination.extend(combinations(sorted_colors, count))
        
        #print(color_combination)
        
        #Convert tuples to list of strings
        color_strings = [''.join(tups) for tups in color_combination]
        
        #print(color_strings)
        
        for color_option in color_options.keys():
            for color_string in color_strings:
                if (len(color_string) == len(color_option)) and set(color_string).issubset(color_option):
                    deck_colors.append(color_option)
                    
        
    except Exception as error:
        print("DeckColors Error: %s" % error)
    
    return deck_colors
    
def AutoColors(deck, color_options, colors_max):
    try:
        deck_colors = next(iter(color_options))
        colors = {}
        if len(deck) > 15:
            deck_colors = DeckColors(deck, color_options, colors_max)[0]
    except Exception as error:
        print("AutoColors Error: %s" % error)
    
    return deck_colors

def CalculateColorAffinity(deck_cards, color_filter, threshold):
    #Identify deck colors  based on the number of high win rate cards
    colors = {}
    try:
        for card in deck_cards:
            gihwr = card["deck_colors"][color_filter]["gihwr"]
            if gihwr > threshold:
                for color in card["colors"]:
                    if color not in colors:
                        colors[color] = 0
                    colors[color] += (gihwr - threshold)
    except Exception as error:
        print("CalculateColorAffinity Error: %s" % error)
    return colors 

def CardFilter(cards, deck, color, color_options, limits, ryanbot, pick_n, pack_n):
    if color == "AI":
        filtered_cards = CardAIFilter(cards, deck, color_options, limits, pick_n, pack_n)
    elif color == "RyanBot":
        filtered_cards = CardRyanBotFilter(cards, deck, limits, ryanbot, pick_n, pack_n)
    else:
        filtered_cards = CardColorFilter(cards, color, limits)
    return filtered_cards

def CardColorFilter(card_list, color, limits):
    try:
        filtered_list = []
        for card in card_list:
            for deck_color in card["deck_colors"].keys():
                if deck_color == color:
                    selected_card = card
                    selected_card["rating_filter"] = CardRating(card["deck_colors"][color], limits[color])
                    selected_card["rating_all"] = CardRating(card["deck_colors"]["All Decks"], limits["All Decks"])
                    filtered_list.append(selected_card)
    except Exception as error:
        print("CardColorFilter Error: %s" % error)
    return filtered_list

def CardRyanBotFilter(pack, deck, limits, ryanbot, pick_n, pack_n):
    try:
        filtered_list = []
        pick_number = ((pack_n - 1) * 14) + pick_n - 1
        for card in pack:
            selected_card = card
            selected_card["rating_all"] = ryanbot.rating(pick_number, selected_card['name'], full_set=True)
            selected_card["rating_filter"] = ryanbot.rating(pick_number, selected_card['name'])
            filtered_list.append(selected_card)
    except Exception as error:
        print("CardRyanBotFilter Error: %s" % error)
    return filtered_list

def CardAIFilter(pack, deck, color_options, limits, pick_n, pack_n):
    try:
        filtered_list = []
        color_limit_start = 15

        pick_number = ((pack_n - 1) * 14) + pick_n - 1
        
        color_count = 3 if pick_number < color_limit_start else 2
        
        deck_colors = DeckColors(deck, color_options, color_count)
        
        for card in pack:
            selected_card = card
            selected_card["rating_all"] = CardRating(selected_card["deck_colors"]["All Decks"], limits["All Decks"])
            
            selected_card["rating_filter"] = CardAIRating(selected_card,
                                                          deck,
                                                          deck_colors,
                                                          limits,
                                                          pick_number)
            filtered_list.append(selected_card)
    except Exception as error:
        print("CardAIFilter Error: %s" % error)
    return filtered_list
     

def RowColorTag(colors):
    row_tag = "goldcard"
    if len(colors) > 1:
        row_tag = "goldcard"
    elif "R" in colors:
        row_tag = "redcard"
    elif "U" in colors:
        row_tag = "bluecard"
    elif "B" in colors:
        row_tag = "blackcard"
    elif "W" in colors:
        row_tag = "whitecard"
    elif "G" in colors:
        row_tag = "greencard"
    return row_tag
    
def DeckColorLimits(cards, color):
    upper_limit = 0
    lower_limit = 100
    try:
        for card in cards:
            gihwr = cards[card]["deck_colors"][color]["gihwr"]
            if gihwr > upper_limit:
                upper_limit = gihwr
            
            if gihwr < lower_limit and gihwr != 0:
                lower_limit = gihwr
    except Exception as error:
        print("DeckColorLimits Error: %s" % error)
    return upper_limit, lower_limit

def CardRating(card_data, limits):
    rating = 0
    winrate = card_data["gihwr"]
    try:
        upper_limit = limits["upper"]
        lower_limit = limits["lower"]
        
        if (winrate != 0) and (upper_limit != lower_limit):
            #Calculate the ALSA bonus
            alsa_bonus = ((9 - card_data["alsa"]) / 8) * 0.5
            
            #Calculate IWD penalty
            iwd_penalty = 0
            
            if card_data["iwd"] < 0:
                iwd_penalty = (max(card_data["iwd"], -10) / 10) * 3
        
            winrate = winrate + iwd_penalty + alsa_bonus
            
            if winrate > limits["upper"]:
                winrate = limits["upper"]
            elif winrate < limits["lower"]:
                winrate = limits["lower"]
            
            rating = ((winrate - lower_limit) / (upper_limit - lower_limit)) * 5.0
            
            rating = round(rating, 1)
            
    except Exception as error:
        print("CardRating Error: %s" % error)
    return rating
    
def CardAIRating(card, deck, deck_colors, limits, pick_number):
    rating = 0
    curve_bonus = 0
    color_bonus = 0
    curve_start = 15
    color_type = "All Decks"
    
    try:
        #Identify color bonus
        color_bonus = ColorBonus(deck, deck_colors, card)
    
        #Identify curve bonus
        if(pick_number >= curve_start):
            curve_bonus = CurveBonus(deck, deck_colors, card, (pick_number - curve_start))
            
            color_type = deck_colors[0] if color_bonus else "All Decks"
            
    
        winrate = card["deck_colors"][color_type]["gihwr"]

        #print("Curve Bonus %.1f" % curve_bonus)
        
        #Calculate card rating
        upper_limit = limits[color_type]["upper"]
        lower_limit = limits[color_type]["lower"]
        
        if (winrate != 0) and (upper_limit != lower_limit):
            #Calculate the ALSA bonus
            alsa_bonus = ((9 - card["deck_colors"][color_type]["alsa"]) / 8) * 0.5
            
            #Calculate IWD penalty
            iwd_penalty = 0
            
            if card["deck_colors"][color_type]["iwd"] < 0:
                iwd_penalty = (max(card["deck_colors"][color_type]["iwd"], -10) / 10) * 3
        
            winrate = winrate + iwd_penalty + alsa_bonus + color_bonus + curve_bonus
            
            if winrate > limits[color_type]["upper"]:
                winrate = limits[color_type]["upper"]
            elif winrate < limits[color_type]["lower"]:
                winrate = limits[color_type]["lower"]
            
            rating = ((winrate - lower_limit) / (upper_limit - lower_limit)) * 5.0
            
            rating = round(rating, 1)
            
    except Exception as error:
        print("CardAIRating Error: %s" % error)
    return rating
    
def DeckColorStats(deck, color):
    creature_count = 0
    noncreature_count = 0

    try:
        creature_cards = DeckColorSearch(deck, color, ["Creature", "Planeswalker"], True, True, False)
        noncreature_cards = DeckColorSearch(deck, color, ["Creature", "Planeswalker"], False, True, False)
        
        creature_count = len(creature_cards)
        noncreature_count = len(noncreature_cards)
        
    except Exception as error:
        print("DeckColorStats Error: %s" % error)
    
    return creature_count, noncreature_count
    
def CardCmcSearch(deck, offset, starting_cmc, cmc_limit, remaining_count):
    cards = []
    unused = []
    try:
        for count, card in enumerate(deck[offset:]):
            card_cmc = card["cmc"]
            
            if card_cmc + starting_cmc <= cmc_limit:
                card_cmc += starting_cmc
                current_offset = offset + count
                current_remaining = int(max(remaining_count - 1, 0))
                if current_remaining == 0:
                    cards.append(card)
                    unused.extend(deck[current_offset + 1:])
                    break
                elif current_offset > (len(deck) - remaining_count):
                    unused.extend(deck[current_offset:])
                    break
                else:
                    current_offset += 1
                    cards, skipped = CardCmcSearch(deck, current_offset, card_cmc, cmc_limit, current_remaining)
                    if len(cards):
                        cards.append(card)
                        unused.extend(skipped)
                        break 
                    else:
                        unused.append(card) 
            else: 
                unused.append(card)
    except Exception as error:
        print("CardCmcSearch Error: %s" % error)
    
    return cards, unused
    
def DeckRating(deck, deck_type, color):
    rating = 0
    try:
        #Combined GIHWR of the cards
        for count, card in enumerate(deck):
            gihwr = card["deck_colors"][color]["gihwr"]
            if gihwr > 50.0:
                rating += gihwr
        
        #Deck contains the recommended number of creatures
        recommended_creature_count = deck_type["recommended_creature_count"]
        filtered_cards = DeckColorSearch(deck, color, ["Creature", "Planeswalker"], True, True, False)
        
        if len(filtered_cards) < recommended_creature_count:
            rating -= 250
        
        #Average CMC of the creatures is below the ideal cmc average
        cmc_average = deck_type["cmc_average"]
        total_cards = len(filtered_cards)
        total_cmc = 0
        
        for card in filtered_cards:
            total_cmc += card["cmc"]
        
        cmc = total_cmc / total_cards
        
        if cmc > cmc_average:
            rating -= 500
            #print("Failed CMC")
        
        
        #Cards fit distribution
        minimum_distribution = deck_type["distribution"]
        distribution = [0, 0, 0, 0, 0, 0, 0]
        for card in filtered_cards:
            index = int(min(card["cmc"], len(minimum_distribution) - 1))
            distribution[index] += 1
            
        for index, value in enumerate(distribution):
            if value < minimum_distribution[index]:
                rating -= 250
                #print("Failed Distribution")
                break
    except Exception as error:
        print("DeckRating Error: %s" % error)
    
    return rating
    
def CopyDeck(deck, sideboard, set_cards, set):
    deck_copy = ""
    starting_index = 0
    total_deck = len(deck)
    card_count = 0
    basic_lands = ["Mountain","Forest","Swamp","Plains","Island"]
    try:
        #Copy Deck
        deck_copy = "Deck\n"
        starting_index = min(set_cards.keys())
        #identify the arena_id for the cards
        for card in deck:
            for index, set_card in enumerate(set_cards):
                if card["name"] in basic_lands:
                    deck_copy += ("%d %s\n" % (card["count"],card["name"]))
                    card_count += 1
                    break;
                elif set_cards[set_card]["name"] == card["name"]:
                    deck_copy += ("%d %s (%s) %d\n" % (card["count"],card["name"],set, (index + 1)))
                    card_count += 1
                    break
        
        #Copy sideboard
        if sideboard != None:
            deck_copy += "\n"
            starting_index = min(set_cards.keys())
            #identify the arena_id for the cards
            for card in sideboard:
                for index, set_card in enumerate(set_cards):
                    if card["name"] in basic_lands:
                        deck_copy += ("%d %s\n" % (card["count"],card["name"]))
                        card_count += 1
                        break;
                    elif set_cards[set_card]["name"] == card["name"]:
                        deck_copy += ("%d %s (%s) %d\n" % (card["count"],card["name"],set, (index + 1)))
                        card_count += 1
                        break
        
    except Exception as error:
        print("CopyDeck Error: %s" % error)
        
    return deck_copy
   
    
def StackCards(cards, color):
    deck = {}
    deck_list = []
    try:
        for card in cards:
            name = card["name"]
            if name not in deck.keys(): 
                deck[name] = {}
                deck[name]["colors"] = card["colors"]
                deck[name]["types"] = card["types"]
                deck[name]["cmc"] = card["cmc"]
                deck[name]["count"] = 1
                deck[name]["name"] = name               
                deck[name]["image"] = card["image"]
                deck[name]["deck_colors"] = {}
                deck[name]["deck_colors"][color] = {}
                deck[name]["deck_colors"][color]["alsa"] = card["deck_colors"][color]["alsa"]
                deck[name]["deck_colors"][color]["iwd"] = card["deck_colors"][color]["iwd"]
                deck[name]["deck_colors"][color]["gihwr"]= card["deck_colors"][color]["gihwr"]
            else:
                deck[name]["count"] += 1
                
        #Convert to list format
        for card in deck:
            deck_list.append(deck[card])
        
    except Exception as error:
        print("StackCards Error: %s" % error)
        
    return deck_list
    
def CardColors(mana_cost):
    colors = []
    try:
        if "B" in mana_cost:
            colors.append("B")
        
        if "G" in mana_cost:
            colors.append("G")
            
        if "R" in mana_cost:
            colors.append("R")
            
        if "U" in mana_cost:
            colors.append("U") 
            
        if "W" in mana_cost:
            colors.append("W")
    except Exception as error:
        print ("CardColors Error: %s" % error)
    return colors
    
#Identify splashable color
def ColorSplash(cards, colors):
    color_affinity = {}
    splash_color = ""
    try:
        # Calculate affinity
        color_affinity = CalculateColorAffinity(cards,colors,65)
        
        # Modify the dictionary to include ratings
        color_affinity = list(map((lambda x : {"color" : x, "rating" : color_affinity[x]}), color_affinity.keys()))
        #print(color_affinity)
        #Remove the current colors
        filtered_colors = color_affinity[:]
        for color in color_affinity:
            if color["color"] in colors:
                filtered_colors.remove(color)
        # Sort the list by decreasing ratings
        filtered_colors = sorted(filtered_colors, key = lambda k : k["rating"], reverse = True)
            
        if len(filtered_colors):
            splash_color = filtered_colors[0]["color"]
    except Exception as error:
        print("ColorSplash Error: %s" % error)   
    return splash_color
    

#Identify the number of lands needed to fill the deck
def ManaBase(deck):
    maximum_deck_size = 40
    combined_deck = []
    mana_types = {"Swamp" : {"color" : "B", "count" : 0},
                  "Forest" : {"color" : "G", "count" : 0},
                  "Mountain" : {"color" : "R", "count" : 0},
                  "Island": {"color" : "U", "count" : 0},
                  "Plains" : {"color" : "W", "count" : 0}}
    total_count = 0
    try:
        number_of_lands = 0 if maximum_deck_size < len(deck) else maximum_deck_size - len(deck)
        
        #Go through the cards and count the mana types
        for card in deck:
            mana_types["Swamp"]["count"] += card["mana_cost"].count("B")
            mana_types["Forest"]["count"] += card["mana_cost"].count("G")
            mana_types["Mountain"]["count"] += card["mana_cost"].count("R")
            mana_types["Island"]["count"] += card["mana_cost"].count("U")
            mana_types["Plains"]["count"] += card["mana_cost"].count("W")
            
        for land in mana_types:
            total_count += mana_types[land]["count"]
        
        #Sort by lowest cound
        mana_types = dict(sorted(mana_types.items(), key=lambda t: t[1]['count']))
        #Add x lands with a distribution set by the mana types
        for index, land in enumerate(mana_types):
            land_count = round((mana_types[land]["count"] / total_count) * number_of_lands, 0)
            #Minimum of 2 lands for a  splash
            if (land_count == 1) and (number_of_lands > 1):
                land_count = 2
                number_of_lands -= 1
            
            if mana_types[land]["count"] != 0:
                card = {"colors" : mana_types[land]["color"], "types" : "Land", "cmc" : 0.0, "name" : land, "count" : land_count}
                combined_deck.append(card) 
            
    except Exception as error:
        print("ManaBase Error: %s" % error)
    #print(combined_deck)
    return combined_deck
    
def SuggestDeck(taken_cards, color_options, limits):
    minimum_creature_count = 13
    minimum_noncreature_count = 6
    maximum_card_count = 23
    ratings_threshold = 500
    colors_max = 3
    used_list = []
    sorted_decks = {}
   
    try:
        #Retrieve configuration
        with open("config.json", "r") as config:
            config_json = config.read()
            config_data = json.loads(config_json)
            minimum_creature_count = config_data["card_logic"]["minimum_creatures"]
            minimum_noncreature_count = config_data["card_logic"]["minimum_noncreatures"]
            ratings_threshold = config_data["card_logic"]["ratings_threshold"]
            deck_builder_type = config_data["card_logic"]["deck_types"]
        #Calculate the base ratings
        filtered_cards = CardColorFilter(taken_cards, "All Decks", limits)
        
        #Identify the top color combinations
        colors = DeckColors(taken_cards, color_options, colors_max)
        
        #print(colors)
        
        filtered_colors = []
        
        #Collect color stats and remove colors that don't meet the minimum requirements
        for color in colors:
            creature_count, noncreature_count = DeckColorStats(filtered_cards, color)
            if((creature_count >= minimum_creature_count) and 
               (noncreature_count >= minimum_noncreature_count) and
               (creature_count + noncreature_count >= maximum_card_count)):
                filtered_colors.append(color)
            
        #print(filtered_colors)
        
        decks = {}
        for color in filtered_colors:
            for type in deck_builder_type.keys():
                deck, sideboard_cards = BuildDeck(deck_builder_type[type], taken_cards, color, limits, minimum_noncreature_count)
                rating = DeckRating(deck, deck_builder_type[type], color)
                if rating >= ratings_threshold:
                    
                    if ((color not in decks.keys()) or 
                        (color in decks.keys() and rating > decks[color]["rating"] )):
                        decks[color] = {}
                        decks[color]["deck_cards"] = StackCards(deck, color)
                        decks[color]["sideboard_cards"] = StackCards(sideboard_cards, color)
                        decks[color]["rating"] = rating
                        decks[color]["type"] = type
                        #print("Color: %s, Rating: %s" % (color, decks[color]["rating"]))
                        decks[color]["deck_cards"].extend(ManaBase(deck))
        
        sorted_colors  = sorted(decks, key=lambda x: decks[x]["rating"], reverse=True)
        #print(sorted_colors)
        for color in sorted_colors:
            sorted_decks[color] = decks[color]
    except Exception as error:
        print("SuggestDeck Error: %s" % error)

    return sorted_decks
    
def BuildDeck(deck_type, cards, color, limits, minimum_noncreature_count):
    minimum_distribution = deck_type["distribution"]
    maximum_card_count = deck_type["maximum_card_count"]
    maximum_deck_size = 40
    cmc_average = deck_type["cmc_average"]
    recommended_creature_count = deck_type["recommended_creature_count"]
    used_list = []
    sideboard_list = cards[:] #Copy by value
    try:
        #filter cards using the correct deck's colors
        filtered_cards = CardColorFilter(cards, color, limits)
        
        #identify a splashable color
        color +=(ColorSplash(filtered_cards, color))
        
        card_colors_sorted = DeckColorSearch(filtered_cards, color, ["Creature", "Planeswalker"], True, True, False)
        card_colors_sorted = sorted(card_colors_sorted, key = lambda k: k["rating_filter"], reverse = True)
        #print(color)
        
        #Identify creatures that fit distribution
        distribution = [0,0,0,0,0,0,0]
        unused_list = []
        used_list = []
        used_count = 0
        used_cmc_combined = 0
        for card in card_colors_sorted:
            index = int(min(card["cmc"], len(minimum_distribution) - 1))
            if(distribution[index] < minimum_distribution[index]):
                used_list.append(card)
                distribution[index] += 1
                used_count += 1
                used_cmc_combined += card["cmc"]
            else:
                unused_list.append(card)
                
                
        #Go back and identify remaining creatures that have the highest base rating but don't push average above the threshold
        unused_cmc_combined = cmc_average * recommended_creature_count - used_cmc_combined
        
        unused_list.sort(key=lambda x : x["rating_filter"], reverse = True)
        
        #Identify remaining cards that won't exceed recommeneded CMC average
        cmc_cards, unused_list = CardCmcSearch(unused_list, 0, 0, unused_cmc_combined, recommended_creature_count - used_count)
        used_list.extend(cmc_cards)
        
        total_card_count = len(used_list)
        
        temp_unused_list = unused_list[:]
        if len(cmc_cards) == 0:
            for count, card in enumerate(unused_list):
                if total_card_count >= recommended_creature_count:
                    break
                    
                used_list.append(card)
                temp_unused_list.remove(card)
                total_card_count += 1
        unused_list = temp_unused_list[:]
            
        card_colors_sorted = DeckColorSearch(filtered_cards, color, ["Instant", "Sorcery","Enchantment","Artifact"], True, True, False)
        card_colors_sorted = sorted(card_colors_sorted, key = lambda k: k["rating_filter"], reverse = True)
        #Add non-creature cards
        for count, card in enumerate(card_colors_sorted):
            if total_card_count >= maximum_card_count:
                break
                
            used_list.append(card)
            total_card_count += 1
            
                
        #Fill the deck with remaining creatures
        for count, card in enumerate(unused_list):
            if total_card_count >= maximum_card_count:
                break
                
            used_list.append(card)
            total_card_count += 1
            

        #Add in special lands if they are on-color, off-color, and they have a card rating above 2.0
        card_colors_sorted = DeckColorSearch(filtered_cards, color, ["Land"], True, True, False)
        card_colors_sorted = sorted(card_colors_sorted, key = lambda k: k["rating_filter"], reverse = True)
        for count, card in enumerate(card_colors_sorted):
            if total_card_count >= maximum_deck_size:
                break
                
            if card["rating_filter"] >= 2.0:    
                used_list.append(card)
                total_card_count += 1
            
        
        #Identify sideboard cards:
        for card in used_list:
            try:
                sideboard_list.remove(card)
            except Exception as error:
                print("%s error: %s" % (card["name"], error))
    except Exception as error:
        print("BuildDeck Error: %s" % error)
    return used_list, sideboard_list
    