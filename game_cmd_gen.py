# 旧代码，完成后需要被禁用，还原指令生成
from game import Game_with_navigator, Game_handle_worldmap
from common_new import logging
import common_new as common

logger = logging.getLogger(__name__)
KITCHENWARES = ['oven', 'stove', 'BBQ']
COOK_COMMAND_RESTRICT = True # True的情况，如果库存和食谱中有相同的物品，才会生成cook命令
CUT_COMMANDS = ['slice', 'chop', 'dice']

def game_state_from_game(game):
    return game

class Game_command_generate_nav(Game_with_navigator):
    def get_admissible_commands(self):
        cook_commands = self.cook_command_generate()
        knife_commands = self.knife_command_generate()
        drop_commands = self.drop_command_generate()
        eat_commands = self.eat_command_generate()
        take_commands = self.take_command_generate()
        open_commands = self.open_command_generate()
        prepare_meal_commands = self.prepare_meal_command_generate()
        go_commands = self.go_command_generate()
        examine_cookbook = self.examine_cookbook_command_generate()
        all_commands = cook_commands + knife_commands + take_commands + drop_commands + \
            open_commands + go_commands + prepare_meal_commands + eat_commands + examine_cookbook
        all_commands = all_commands + self.navigate_command_generate()
        return all_commands
    
    def filter_enetities_in_ingredients(self, candidate_entities = None):
        if self.recipe == '':
            logger.debug('No recipe found in game state')
            return []
        results = []
        game_state = game_state_from_game(self)
        ingredients = game_state.ingredients_from_recipe()
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, ingredients):
                results.append(entity)
        return results
    
    def filter_enetities_in_inventory(self, candidate_entities = None):
        entities = []
        game_state = game_state_from_game(self)
        inventory = game_state.inventory_clean()
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, inventory):
                entities.append(entity)
        return entities
    
    def entities_in_description(self, candidate_entities = None, use_raw_description = False):
        entities = []
        game_state = game_state_from_game(self)
        desc = game_state.description_clean() if not use_raw_description else self.info['description']
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, desc):
                entities.append(entity)
        return entities
    
    def kitchenware_in_description(self):
        exist_kitchenware = self.entities_in_description(candidate_entities=KITCHENWARES)
        return exist_kitchenware
    
    def drop_command_generate(self):
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        if len(entities_in_inventory) == 0:
            logger.debug('No entities found in inventory, no need to generate drop commands')
            return []
        drop_commands = []
        for entity in entities_in_inventory:
            drop_commands.append(f'drop {entity}')
        return drop_commands

    def knife_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate cut commands')
            return []
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        if 'knife' not in entities_in_inventory:
            logger.debug('No knife found in inventory, no need to generate cut commands')
            return []
        foods_in_inventory = [entity for entity in entities_in_inventory if entity != 'knife']
        if COOK_COMMAND_RESTRICT:
            foods_in_inventory = self.filter_enetities_in_ingredients(foods_in_inventory)
        retults = []
        for food in foods_in_inventory:
            for cut_command in CUT_COMMANDS:
                retults.append(f'{cut_command} {food} with knife')
        return retults

    def cook_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate cook commands')
            return []
        exist_kitchenware = self.kitchenware_in_description()
        if len(exist_kitchenware) == 0:
            # logger.debug('No kitchenware found in description, no need to generate cook commands')
            return []
        cook_commands = []
        entities = self.filter_enetities_in_inventory(self.info['entities'])
        if COOK_COMMAND_RESTRICT:
            entities = self.filter_enetities_in_ingredients(entities)
        for entity in entities:
            for ware in exist_kitchenware:
                cook_commands.append(f'cook {entity} with {ware}')
        return cook_commands
    
    def eat_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate eat commands')
            return []
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        if 'meal' in entities_in_inventory:
            return ['eat meal']
        else:
            return []
    
    def take_command_generate(self):
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate take commands')
            return []
        entities_to_take = self.info['entities']
        if COOK_COMMAND_RESTRICT:
            entities_to_take = self.filter_enetities_in_ingredients(self.info['entities'])
            entities_to_take = entities_to_take + ['knife', 'meal']
        # 判断环境中是否有这些物品
        entities_to_take = self.entities_in_description(candidate_entities=entities_to_take)
        return [f'take {entity}' for entity in entities_to_take]
    
    def open_command_generate_old(self):
        # 判断环境中是否有这些物品
        # 清除frosted-glass door中的横杠
        entities = self.entities_in_description(candidate_entities=self.info['entities'], use_raw_description=True)
        entities_to_open = []
        for entity in entities:
            if entity.endswith('door') or is_openable_entity(entity):
                entities_to_open.append(entity)
        return [f'open {entity}' for entity in entities_to_open]

    def open_command_generate(self):
        # NOTE: 2026.5.26 直接从admissible comands中找open命令
        cmds = common.filter_commands_default(self.info['admissible_commands'])
        return [cmd for cmd in cmds if cmd.startswith('open ')]

    
    def prepare_meal_command_generate(self):
        # 判断，当身处厨房，有食谱，且背包里的物品和食谱中的物品相同，才会生成prepare meal命令
        if self.recipe == '':
            logger.debug('No recipe found in game state, no need to generate prepare meal commands')
            return []
        game_state = game_state_from_game(self)
        room = game_state.room
        if room.lower() != 'kitchen':
            logger.debug('Not in kitchen, no need to generate prepare meal commands')
            return []
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        entities_in_recipe = self.filter_enetities_in_ingredients(self.info['entities'])
        for entity in entities_in_recipe:
            if entity not in entities_in_inventory:
                logger.debug(f'Entity {entity} not in inventory, no need to generate prepare meal commands')
                return []
        return ['prepare meal']
    
    def go_command_generate_old(self):
        directions = ['north', 'south', 'east', 'west']
        filtered_directions = self.entities_in_description(candidate_entities=directions)
        if len(filtered_directions) == 0:
            logger.debug('No directions found in description, no need to generate go commands')
            return []
        go_commands = ['go ' + direction for direction in filtered_directions]
        return go_commands
    
    def go_command_generate(self):
        # NOTE: 2026.5.26 直接从admissible comands中找go命令
        cmds = common.filter_commands_default(self.info['admissible_commands'])
        return [cmd for cmd in cmds if cmd.startswith('go ')]
    
    def examine_cookbook_command_generate(self):
        if self.recipe != '':
            return []
        entities_in_description = self.entities_in_description(candidate_entities=self.info['entities'])
        if 'cookbook' not in entities_in_description:
            logger.debug('No cookbook found in description, no need to generate examine cookbook commands')
            return []
        return ['examine cookbook']


class Game_command_generate_vanilla(Game_command_generate_nav):
    def get_admissible_commands(self):
        cook_commands = self.cook_command_generate()
        knife_commands = self.knife_command_generate()
        drop_commands = self.drop_command_generate()
        eat_commands = self.eat_command_generate()
        take_commands = self.take_command_generate()
        open_commands = self.open_command_generate()
        prepare_meal_commands = self.prepare_meal_command_generate()
        go_commands = self.go_command_generate()
        examine_cookbook = self.examine_cookbook_command_generate()
        all_commands = cook_commands + knife_commands + take_commands + drop_commands + \
            open_commands + go_commands + prepare_meal_commands + eat_commands + examine_cookbook
        # all_commands = all_commands + self.navigate_command_generate() # NOTE: vanilla模型不需要navigate命令
        return all_commands  

def default_game():
    return Game_command_generate_nav(f'{common.GAME_BASE_PATH}/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')