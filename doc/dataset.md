应该是这样的，首先我们将available commands先过滤一遍，然后理论上game在执行指令列表的同时能够生成navigate指令

所以最终要获取的指令列表我们可以选带navigate指令版本和不带的。或者用两个类来代理不同的available commands

现在的game_with_navigator是继承game_bert_filter_command的，我们要删掉中间环节直接将他连接到game_with_worldmap

现在我们要生成带navigation指令的数据集


## memo of `dataset_create.py`

1. `extract_walkthrough_dataset_with_navigator`函数就是制作数据集的逻辑所在。
2. `get_clean_clean_walkthrough`函数会先跑一次游戏，然后确保游戏能正常结束。
3. 同上函数，我们会先将`take apple from fridge`替换成`take apple`
4. 同上函数，如果是admissable actions中不存在的指令，我们也跳过
5. 

