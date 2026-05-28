import agent_navigator as agent
import game_cmd_gen
agent.GAME_INIT_FUNC = game_cmd_gen.Game_command_generate
print('set GAME_INIT_FUNC to Game_command_generate')