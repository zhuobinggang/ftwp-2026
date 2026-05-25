# 使用Game_handle_worldmap作为基础类，能够执行navigate命令，但是不生成navigate指令在available commands中
# 使用Game_with_navigator作为基础类，能够执行navigate命令，并且生成navigate指令在available commands中
import random
import common_new as common
import textworld.gym
from textworld import EnvInfos, gym
from functools import lru_cache
from recordclass import recordclass
import logging

logger = logging.getLogger('game.py')
dbg = logger.debug

MAX_STEP = 100

# 重新实现game
def init_env(game_file):
    requested_infos = EnvInfos(description=True, inventory=True,
                               admissible_commands=True, objective=False,
                               # verbs=True, command_templates=True,
                               entities=True, max_score=True, won=True, score=True,
                               moves = True,
                               lost=True, extras=["walkthrough"]) # 注意，取不到recipe，只能从obs中获取
    env_id = textworld.gym.register_games([game_file], requested_infos, max_episode_steps = MAX_STEP)
    env = gym.make(env_id)
    return env

TestResult = recordclass('TestResult', 'step score max_score info')

class Game:
    def __init__(self, game_path, need_init_env = True):
        self.game_path = game_path
        self.env = init_env(game_path) if need_init_env else None
        self.obs, self.info = None, None
        self.reward, self.done = 0, False
        self.room = ''
        self.inventory_raw = ''
        self.description_raw = ''
    def reset(self):
        self.obs, self.info = self.env.reset()
        self.room = common.extract_room_name(self.info['description'])
        self.inventory_raw = self.info['inventory']
        self.description_raw = self.info['description']
        return self.obs, self.info
    def act(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        self.room = common.extract_room_name(self.info['description']) # 每一步都更新房间信息
        self.inventory_raw = self.info['inventory'] # 每一步都更新库存信息
        self.description_raw = self.info['description'] # 每一步都更新描述信息
        return self.obs, self.reward, self.done, self.info
    def clean_walkthrough(self):
        return common.filter_commands_default(self.info['extra.walkthrough'])
    def inventory_clean(self):
        if self.inventory_raw == '':
            return ''
        else:
            return common.handle_inventory_text(self.inventory_raw)
    def description_clean(self):
        return common.description_simplify(self.description_raw)
    def get_admissible_commands(self):
        return common.filter_commands_default(self.info['admissible_commands'])
    def available_commands_text(self):
        return common.actions_to_list_number(self.get_admissible_commands())


class Fake_model:
    def __init__(self):
        self.counter = 0
    def predict(self, obs, info):
        # 这里需要根据obs和info来选择动作
        # 这里简单返回一个随机动作
        action = info['extra.walkthrough'][self.counter]
        self.counter += 1
        if self.counter >= len(info['extra.walkthrough']):
            self.counter = 0
        return action
    def eval(self):
        pass
    def cuda(self):
        pass

# ============

class Game_with_history(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_obs_pairs = []
    def act(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        self.action_obs_pairs.append((action, self.obs))
        return self.obs, self.reward, self.done, self.info
    def clean_action_obs_pairs(self):
        return [clean_action_obs(action, obs) for action, obs in self.action_obs_pairs]
    def action_history(self, history_window = 100, seperator='>', no_action_text=''):
        action_obs_pairs = self.clean_action_obs_pairs()
        action_history_text = common.action_obs_pairs_to_history(action_obs_pairs, seperator=seperator, no_action_text=no_action_text, history_window = history_window)        
        return action_history_text
    def action_history_simple(self, history_window = 5, seperator='>', no_action_text='empty'):
        action_obs_pairs = self.clean_action_obs_pairs()
        action_history_text = common.action_obs_pairs_to_history_simple(action_obs_pairs, seperator=seperator, no_action_text=no_action_text, history_window = history_window)        
        return action_history_text
    
class Game_handle_recipe(Game_with_history):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipe_raw = ''
        self.recipe = ''
        self.obs_raw = ''
    def act(self, action):
        self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
        obs = self.obs_raw
        # obs simplify
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
        self.action_obs_pairs.append((action, obs)) # 不在这里处理，而是放到game_state中处理
        self.obs = obs
        return self.obs, self.reward, self.done, self.info
    def recipe_clean(self):
        return self.recipe
    
    
# NOTE: 在移动命令被执行后，obs改为prev_room to current_room。这样能够给模型一个直观的记忆。因为在prompt中没有上一个房间的信息，应该很有帮助。
class Game_move_action_augment(Game_handle_recipe):
    def act(self, action):
        # 这里处理移动命令
        if action.startswith('go'):
            prev_room = common.extract_room_name(self.info['description'])
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            current_room = common.extract_room_name(self.obs_raw)
            obs = f'From {prev_room} to {current_room}.'
        else:
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            obs = self.obs_raw
        # obs simplify
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
        self.action_obs_pairs.append((action, obs)) # 不在这里处理，而是放到game_state中处理
        self.obs = obs
        return self.obs, self.reward, self.done, self.info
    
def bfs_search(start, goal, worldMap):
    queue = [(start, [])]
    visited = set()
    while queue:
        current_room, path = queue.pop(0)
        if current_room == goal:
            return path
        visited.add(current_room)
        for direction, next_room in worldMap[current_room].items():
            if next_room not in visited:
                queue.append((next_room, path + [direction]))
    return []

def directions_to_action(directions):
    return ['go ' + direction for direction in directions]

class Game_handle_worldmap(Game_move_action_augment):
    def __init__(self, game_path):
        super().__init__(game_path)
        self.worldMap = {}
        self.itemMap = {}
    def navigate_to_item(self, itemName):
        item = self.itemMap[itemName]
        if item['room'] == 'inventory':
            logger.error(f'Item {itemName} is in inventory, cannot navigate to it.')
            return []
        target_room = item['room']
        if target_room not in self.worldMap:
            logger.error(f'Item {itemName} is in unknown room {target_room}.')
            return []
        return self.navigate_to_room(target_room)
    def navigate_to_room(self, target_room):
        current_room = common.extract_room_name(self.info['description'])
        path = bfs_search(current_room, target_room, self.worldMap)
        if len(path) == 0:
            logger.error(f'Cannot find path from {current_room} to {target_room}.')
            return []
        else:
            return directions_to_action(path)
    def reset(self):
        self.obs, self.info = super().reset()
        self.worldMap[self.room] = {}
        return self.obs, self.info
    def act(self, action):
        action = action.strip()
        # 这里处理移动命令
        if action.startswith('go'):
            prev_room = common.extract_room_name(self.info['description'])
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            current_room = common.extract_room_name(self.info['description'])
            obs = f'From {prev_room} to {current_room}.'
            if prev_room == current_room:
                logger.warning(f'Action {action} should not happen, prev_room == current_room: {prev_room}')
            elif True: # 更新worldMap 
                if prev_room not in self.worldMap:
                    self.worldMap[prev_room] = {}
                if current_room not in self.worldMap:
                    self.worldMap[current_room] = {}
                direction = action.split()[1]
                op_direction = common.get_opposite_direction(direction)
                self.worldMap[prev_room][direction] = current_room
                self.worldMap[current_room][op_direction] = prev_room
        elif action.startswith('navigate to '):
            # logger.warning(f'{action}')
            prev_room = common.extract_room_name(self.info['description'])
            entity_or_room = action.replace('navigate to ', '')
            path = []
            if entity_or_room in self.itemMap:
                target_room = self.itemMap[entity_or_room]['room']
                path = self.navigate_to_room(target_room)
            elif entity_or_room in self.worldMap:
                target_room = entity_or_room
                path = self.navigate_to_room(target_room)
            for temp_action in path: # 导航到目标房间
                self.obs_raw, self.reward, self.done, self.info = self.env.step(temp_action)
            obs = f'Navigate from {prev_room} to {target_room}.'
            logger.debug(f'{obs}, path: {path}')
        else:
            self.obs_raw, self.reward, self.done, self.info = self.env.step(action)
            obs = self.obs_raw
        # obs simplify
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
        self.action_obs_pairs.append((action, obs)) # 不在这里处理，而是放到game_state中处理
        self.obs = obs
        # 更新self.room
        self.room = common.extract_room_name(self.info['description'])
        if True: # NOTE: 每一步根据环境描述来更新itemMap
            # 每一步根据recipe & 环境描述来更新itemList。item包含字段：room。
            entities = self.info['entities']
            # 去除east, west, north, south等方向词，避免误匹配
            entities = [entity for entity in entities if entity not in common.DIRECTIONS]
            for entity in entities:
                if common.whole_word_inside(entity, self.info['description']):
                    if entity not in self.itemMap:
                        self.itemMap[entity] = {'room': ''}
                    if self.room != self.itemMap[entity]['room']:
                        # logger.debug(f'Update itemMap: {entity} from {self.itemMap[entity]["room"]} to {room_name}')
                        pass
                    self.itemMap[entity]['room'] = self.room
                if common.whole_word_inside(entity, self.info['inventory']):
                    if entity not in self.itemMap:
                        self.itemMap[entity] = {'room': ''}
                    self.itemMap[entity]['room'] = 'inventory'
        return self.obs, self.reward, self.done, self.info

# NOTE: 2026.5.19 直接继承Game_handle_worldmap
class Game_with_navigator(Game_handle_worldmap):
    def filter_enetities_in_ingredients(self, candidate_entities = None):
        if self.recipe == '':
            logger.debug('No recipe found in game state')
            return []
        results = []
        ingredients = common.ingredients_from_recipe(self.recipe) # 在act中已经清理过了
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, ingredients):
                results.append(entity)
        return results
    def navigate_command_generate(self):
        if self.recipe == '':
            # logger.debug('No recipe found in game state, no need to generate navigate commands')
            return []
        entities = self.filter_enetities_in_ingredients(self.info['entities'])
        entities += ['knife']
        entities += common.KITCHENWARES
        commands = []
        current_room = common.extract_room_name(self.info['description'])
        for entity in entities:
            if entity in self.itemMap:
                target_room = self.itemMap[entity]['room']
                if target_room not in ['inventory', '', current_room]:
                    commands.append(f'navigate to {entity}')
        return commands
    def extra_commands_hook(self):
        return []
    def get_admissible_commands(self):
        all_commands = super().get_admissible_commands()
        all_commands += self.navigate_command_generate()
        all_commands += self.extra_commands_hook()
        #if common.COMMAND_LIST_SHUFFLE:
        #    random.shuffle(all_commands) # NOTE: 2025.5.5 打乱以提高模型的泛化能力 NOTE: 2026.5.24 只在数据集生成时打乱
        return all_commands

def default_game():
    return Game_with_navigator(f'{common.GAME_BASE_PATH}/valid/tw-cooking-recipe1+cook+cut+drop+go6-M2qEFeOXcol3H1ql.ulx')

def test_default_game():
    game = default_game()
    obs, info = game.reset()
    print(obs)
    print(info)
    game.act('drop red onion')
    game.act('go east')
    game.act('examine cookbook')
    print(game.filtered_available_commands())
    return game

@lru_cache(maxsize=128) # 一个episode最多为100步，因此128足够了
def clean_action_obs(action, obs):
    ACT, OBS = action, obs
    if action == 'examine cookbook' and common.is_recipe_feedback(obs):
        OBS = 'recipe got!'
    elif common.is_description_feedback(obs): # NOTE: 如果使用Game_move_action_augment的话，obs会是"From room1 to room2."，不会进入这个分支
        room_name = common.extract_room_name(obs)
        OBS = f'you entered {room_name}.'
    OBS = ' '.join(OBS.split()).strip()
    return ACT, OBS

# 包含了所有需要的信息，不包含textworld环境细节
class Game_state(Game_handle_recipe):
    def __init__(self):
        game_path = None
        super().__init__(game_path, need_init_env=False)
    def __str__(self):
        return f'Game_state(\nroom={self.room}\ndescription={self.description_clean()}\nrecipe={self.recipe_clean()}\ninventory={self.inventory_clean()}\naction_obs_pairs={self.action_history()}\nadmissible_commands={self.available_commands_text()})'

class Game_state_clean(Game_state):
    def __init__(self):
        super().__init__()
        self.recipe_good = ''
        self.inventory_good = ''
        self.description_good = ''
        self.action_obs_pairs_good = []
        self.available_commands_good = []
    def recipe_clean(self):
        return self.recipe_good
    def inventory_clean(self):
        return self.inventory_good
    def description_clean(self):
        return self.description_good
    def filtered_available_commands(self):
        return self.available_commands_good
    def get_admissible_commands(self):
        return self.available_commands_good



def test_game(game: Game_handle_worldmap, model = Fake_model(), max_step = 100, need_print = False):
    # import game_for_llm
    # max_step = 50 # 2025.8.28 实验用，实验结束后删除
    # dbg('Testing: Model eval on, model cuda on.')
    if model.training:
        model.eval()
        dbg('Model eval on.')
    if not next(model.parameters()).is_cuda:
        model.cuda()
        dbg('Model cuda on.')
    obs, info = game.reset()
    counter = 0
    final_action = ''
    while counter < max_step:
        action = model.predict(game)
        prev_moves = game.info['moves']
        obs, reward, done, info = game.act(action)
        if need_print:
            print(f'{action} => {obs}')
        current_moves = info['moves']
        counter += max(1, current_moves - prev_moves) # 考虑到可能的多步行动（比如高级命令）
        final_action = action
        if done:
            break
    # result = (counter, info['score'], info['max_score'], info)
    logger.debug(f'Game done: {info["score"]} / {info["max_score"]}, steps {counter}, won: {info["won"]}, lost: {info["lost"]}, path: {game.game_path}')
    if info['lost']:
        logger.warning(f'Game lost: final action: {final_action}')
    result = TestResult(counter, info['score'], info['max_score'], info)
    return result