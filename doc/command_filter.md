## 记录一下指令过滤的规则

1. 过滤所有以['examine', 'close', 'eat', 'look', 'inventory', 'drink', 'put', 'insert']开头的指令
2. 但是在上述过滤阶段，保留['examine cookbook', 'eat meal']
3. 将所有take指令精简： take apple from fridge -> take apple
4. cook指令：只有在检查cookbook，且配料中包含entity的时候才保留cook entity指令。（加到列表后面）
5. take指令：库存满的情况下引擎不会生成任何take指令，但是我们为knife和所有出现在ingredients中的实体生成take指令（加到列表前面）
6. take指令：没有食谱的情况下不生成任何take指令，有食谱的情况下只对配料和knife保留take指令。（加到列表后面）
7. prepare meal指令： 只有在厨房且当食谱中出现的食材都在库存中时才保留prepare meal命令。（加到列表后面）
8. examine cookbook指令： 在检查过cookbook之后不需要再提供。（加到列表后面）

## 指令过滤和指令合成的区别

1. 在指令过滤中，slice（或者dice或者chop）过的物品不会再提供相同指令，但是指令合成会再次提供
2. 同上，cook指令也不会二次提供