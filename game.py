# 使用Game_handle_worldmap作为基础类，能够执行navigate命令，但是不生成navigate指令在available commands中
# 使用Game_with_navigator作为基础类，能够执行navigate命令，并且生成navigate指令在available commands中
import random
import common_new as common
import textworld.gym
from textworld import EnvInfos, gym
from functools import lru_cache
from recordclass import recordclass
import logging
import copy

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
    def before_act_hook(self, action):
        pass
    def act(self, action): # obs无更改
        action = action.strip()
        self.before_act_hook(action)
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
    def accumulated_score(self):
        return self.info['score']


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
    def after_act_hook(self, action, obs):
        # This method can be overridden by subclasses to modify action-obs pairs
        return action, obs.replace('\n', ' ').strip() # 默认的hook是把obs中的换行符替换成空格，并去除首尾空格
    def act(self, action, do_not_append=False): # obs无更改
        self.obs, self.reward, self.done, self.info = super().act(action)
        new_action, new_obs = self.after_act_hook(action, self.obs)
        if not do_not_append:
            self.action_obs_pairs.append((new_action, new_obs)) # TODO: add hook
        return self.obs, self.reward, self.done, self.info
    def action_history(self, history_window = 100, seperator='>', no_action_text=''):
        action_obs_pairs = self.action_obs_pairs
        action_history_text = common.action_obs_pairs_to_history(action_obs_pairs, seperator=seperator, no_action_text=no_action_text, history_window = history_window)        
        return action_history_text
    def action_history_simple(self, history_window = 5, seperator='>', no_action_text='empty'):
        action_obs_pairs = self.action_obs_pairs
        action_history_text = common.action_obs_pairs_to_history_simple(action_obs_pairs, seperator=seperator, no_action_text=no_action_text, history_window = history_window)        
        return action_history_text
    
class Game_handle_recipe(Game_with_history):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipe_raw = ''
        self.recipe = ''
        self.obs_raw = ''
    def after_act_hook(self, action, obs):
        # This method can be overridden by subclasses to modify action-obs pairs
        action, new_obs = super().after_act_hook(action, obs)
        if action == 'examine cookbook' and common.is_recipe_feedback(obs):
            self.recipe_raw = common.extract_recipe(obs, need_clean=False)
            self.recipe = common.extract_recipe(self.recipe_raw, need_clean=True)
            new_obs = 'recipe got!'
        return action, new_obs
    def recipe_clean(self):
        return self.recipe
    def ingredients_from_recipe(self):
        return common.ingredients_from_recipe(self.recipe_clean())
    def filter_enetities_in_inventory(self, candidate_entities = None):
        entities = []
        inventory = self.inventory_clean()
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, inventory):
                entities.append(entity)
        return entities
    def filter_enetities_in_ingredients(self, candidate_entities = None):
        if self.recipe == '':
            logger.debug('No recipe found in game state')
            return []
        results = []
        ingredients = self.ingredients_from_recipe()
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, ingredients):
                results.append(entity)
        return results
    def filter_cook_commands(self, cmds):
        # NOTE: 只有食谱中出现的食材才保留cook命令
        cook_cmds = [cmd for cmd in cmds if cmd.startswith('cook ')]
        other_cmds = [cmd for cmd in cmds if not cmd.startswith('cook ')]
        if self.recipe == '': # 没有食谱的情况下不cook任何东西
            return other_cmds
        entities_in_recipe = self.filter_enetities_in_ingredients(self.info['entities'])
        entities = [common.extract_cook_command_entity(cook_cmd) for cook_cmd in cook_cmds]
        command_entity_pairs = list(zip(cook_cmds, entities))
        for cmd, entity in command_entity_pairs:
            if entity in entities_in_recipe:
                other_cmds.append(cmd)
        return other_cmds
    def filter_take_commands(self, cmds):
        # NOTE: 只有食谱中出现的食材才保留take命令
        take_cmds = [cmd for cmd in cmds if cmd.startswith('take ')]
        other_cmds = [cmd for cmd in cmds if not cmd.startswith('take ')]
        if self.recipe == '': # 没有食谱的情况下不拿任何东西
            return other_cmds
        entities_in_recipe = self.filter_enetities_in_ingredients(self.info['entities'])
        entities_in_recipe += ['knife']
        entities_in_recipe = set(entities_in_recipe)
        entities_can_take = set([take_cmd.replace('take ', '').strip() for take_cmd in take_cmds])
        entities_can_take = entities_can_take & entities_in_recipe
        for entity in entities_can_take:
            other_cmds.append('take ' + entity) 
        return other_cmds
    def filter_entities_in_description(self, candidate_entities = None, use_raw_description = False):
        entities = []
        desc = self.description_clean() if not use_raw_description else self.info['description']
        candidate_entities = candidate_entities if candidate_entities is not None else self.info['entities']
        for entity in candidate_entities:
            if common.whole_word_inside(entity, desc):
                entities.append(entity)
        return entities
    def try_add_take_commands(self, cmds):
        take_cmds = [cmd for cmd in cmds if cmd.startswith('take ')]
        if len(take_cmds) > 0:
            return cmds # 已经有take命令了就不添加了。 NOTE: 在加入这一判断之前可能生成了重复的take命令；问题应该不大，只是训练时候可能会有重复的take命令被选中，另一方面推理时候可能耗时长一些--- IGNORE ---
        else:
            # NOTE: 在库存满了的时候不能take，但是我们希望能继续生成用于负反馈
            # 出现在description和recipe中的实体，再包括knife都可以被take
            entities_in_recipe = self.filter_enetities_in_ingredients(self.info['entities'])
            entities_in_recipe += ['knife']
            entities_in_description = self.filter_entities_in_description(self.info['entities'])
            entities_can_take = set(entities_in_recipe) & set(entities_in_description)
            take_commands_added = ['take ' + entity for entity in entities_can_take]
            # print(f'Added take commands: {take_commands_added}')
            return take_commands_added + cmds

    def filter_prepare_meal_command(self, cmds):
        # NOTE: 只有当食谱中出现的食材都在库存中时才保留prepare meal命令
        if 'prepare meal' not in cmds:
            return cmds
        cmds_without_prepare_meal = cmds.copy()
        cmds_without_prepare_meal.remove('prepare meal')
        if self.recipe == '': # 没找到食谱
            return cmds_without_prepare_meal
        if self.room.lower() != 'kitchen': # 不在厨房
            return cmds_without_prepare_meal
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        entities_in_recipe = self.filter_enetities_in_ingredients(self.info['entities'])
        for entity in entities_in_recipe:
            if entity not in entities_in_inventory: # 食谱中出现的食材不在库存中
                logger.debug(f'Entity {entity} not in inventory, no need to generate prepare meal commands')
                return cmds_without_prepare_meal
        return cmds_without_prepare_meal + ['prepare meal']
    def filter_examine_cookbook_command(self, cmds):
        if 'examine cookbook' not in cmds:
            return cmds
        cmds_without_examine_cookbook = cmds.copy()
        cmds_without_examine_cookbook.remove('examine cookbook')
        if self.recipe != '': # 已经找到食谱了
            return cmds_without_examine_cookbook
        return cmds_without_examine_cookbook + ['examine cookbook']
    def try_add_eat_meal_if_necessary(self, cmds):
        if 'eat meal' in cmds:
            return cmds
        entities_in_inventory = self.filter_enetities_in_inventory(self.info['entities'])
        if 'meal' in entities_in_inventory:
            print('生成了eat meal指令……说明数据集有问题')
            return cmds + ['eat meal']
        else:
            return cmds
    def get_admissible_commands(self):
        cmds = super().get_admissible_commands()
        # TODO: 对于cook指令，只有出现在ingredients中才保留
        cmds = self.filter_cook_commands(cmds)
        cmds = self.try_add_take_commands(cmds) # 在库存满了的时候不能take，但是我们希望能继续生成用于负反馈
        cmds = self.filter_take_commands(cmds)
        cmds = self.filter_prepare_meal_command(cmds)
        cmds = self.filter_examine_cookbook_command(cmds)
        cmds = self.try_add_eat_meal_if_necessary(cmds)
        return cmds
    
# NOTE: 在移动命令被执行后，obs改为prev_room to current_room。这样能够给模型一个直观的记忆。因为在prompt中没有上一个房间的信息，应该很有帮助。
class Game_move_action_augment(Game_handle_recipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_room = ''
    def before_act_hook(self, action):
        if action.startswith('go'):
            self.prev_room = common.extract_room_name(self.info['description'])
    def after_act_hook(self, action, obs): # after act hook
        action, new_obs = super().after_act_hook(action, obs)
        if action.startswith('go'):
            prev_room = self.prev_room
            current_room = self.room # setted in Game.act
            new_obs = f'From {prev_room} to {current_room}.'
        return action, new_obs
    
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
        self.itemMap = {}
        self.worldMap[self.room] = {} # 必须在reset的时候就把初始房间加入worldMap，不然会报错
        self.update_item_map() # 根据初始状态更新itemMap
        return self.obs, self.info
    def update_item_map(self):
        # 每一步根据recipe & 环境描述来更新itemList。item包含字段：room。
        entities = self.info['entities']
        # 去除east, west, north, south等方向词，避免误匹配
        entities = [entity for entity in entities if entity not in common.DIRECTIONS]
        for entity in entities:
            if common.whole_word_inside(entity, self.info['description']):
                if entity not in self.itemMap:
                    self.itemMap[entity] = {'room': ''}
                self.itemMap[entity]['room'] = self.room # setted in Game.act
            if common.whole_word_inside(entity, self.info['inventory']):
                if entity not in self.itemMap:
                    self.itemMap[entity] = {'room': ''}
                self.itemMap[entity]['room'] = 'inventory'
    def after_act_hook(self, action, obs): # after move action, update worldMap and itemMap
        action, obs = super().after_act_hook(action, obs)
        if action.startswith('go'):
            current_room = self.room # setted in Game.act
            prev_room = self.prev_room # setted in Game_move_action_augment.before_act_hook
            if True: # 更新worldMap
                if prev_room not in self.worldMap:
                    self.worldMap[prev_room] = {}
                if current_room not in self.worldMap:
                    self.worldMap[current_room] = {}
                direction = action.split()[1]
                op_direction = common.get_opposite_direction(direction)
                self.worldMap[prev_room][direction] = current_room
                self.worldMap[current_room][op_direction] = prev_room
        self.update_item_map() # Always update itemMap
        return action, obs
    def act(self, action): # 要代理navigate命令
        # 代理navigate命令，循环act goes
        if action.startswith('navigate to '):
            # logger.warning(f'{action}')
            prev_room = self.room # 目前还没act，所以self.prev_room还是上一个房间
            entity_or_room = action.replace('navigate to ', '')
            path = []
            if entity_or_room in self.itemMap:
                target_room = self.itemMap[entity_or_room]['room']
                path = self.navigate_to_room(target_room)
            elif entity_or_room in self.worldMap:
                target_room = entity_or_room
                path = self.navigate_to_room(target_room)
            if prev_room == target_room:
                print(f'Already in {target_room}, no need to execute {action}. 可能是循环导航。')
                logger.warning(f'Already in {target_room}, no need to execute {action}.可能是循环导航。')
            else:
                for temp_action in path: # 导航到目标房间
                    self.obs_raw, self.reward, self.done, self.info = super().act(temp_action, do_not_append=True) # 执行导航指令不记录action_obs_pair
                navigate_obs = f'Navigate from {prev_room} to {target_room}.'
                self.action_obs_pairs.append((action, navigate_obs)) # 记录导航指令的action_obs_pair
                logger.debug(f'{navigate_obs}, path: {path}')
        else: # 否则正常执行
            self.obs_raw, self.reward, self.done, self.info = super().act(action)
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
        current_room = self.room
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
        self.available_commands_good = []
        self.accumulated_score_good = 0
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
    def accumulated_score(self):
        return self.accumulated_score_good
    
class Game_state_clean_with_worldmap(Game_state_clean):
    def __init__(self):
        super().__init__()
        self.worldMap = {}
        self.itemMap = {}

def game_state_from_game(game: Game_handle_worldmap, need_worldmap = True, need_action_obs_pairs = True):
    game_state = Game_state_clean_with_worldmap()
    game_state.recipe_good = game.recipe_clean()
    game_state.inventory_good = game.inventory_clean()
    game_state.description_good = game.description_clean()
    game_state.available_commands_good = game.get_admissible_commands().copy()
    if need_worldmap:
        game_state.worldMap = copy.deepcopy(game.worldMap)
        game_state.itemMap = copy.deepcopy(game.itemMap)
    game_state.room = game.room
    if need_action_obs_pairs:
        game_state.action_obs_pairs = game.action_obs_pairs.copy()
    return game_state

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
        action = model.predict(game_state_from_game(game))
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