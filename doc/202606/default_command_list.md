## 不对指令列表进行奇怪过滤的实验结果 20260606 cta_nav_cmdraw

```log
Evaluating 0/3 checkpoint: ./checkpoints/cta_nav_cmdraw/roberta_navigator_20260605_100133_024368_best.pth
Stop epoch: 1, valid score: 0.9849094567404426
Validation on test norm_score: 0.9251944943147815
Checkpoint: ./checkpoints/cta_nav_cmdraw/roberta_navigator_20260605_100133_024368_best.pth, test score: 0.9251944943147815, average_step: 20.19455252918288
Evaluating 1/3 checkpoint: ./checkpoints/cta_nav_cmdraw/roberta_navigator_20260605_100133_030437_best.pth
Stop epoch: 4, valid score: 0.9919517102615694
Validation on test norm_score: 0.9539198084979055
Checkpoint: ./checkpoints/cta_nav_cmdraw/roberta_navigator_20260605_100133_030437_best.pth, test score: 0.9539198084979055, average_step: 20.71011673151751
Evaluating 2/3 checkpoint: ./checkpoints/cta_nav_cmdraw/roberta_navigator_20260605_110915_021471_best.pth
Stop epoch: 2, valid score: 1.0
Validation on test norm_score: 0.9230999401555955
Checkpoint: ./checkpoints/cta_nav_cmdraw/roberta_navigator_20260605_110915_021471_best.pth, test score: 0.9230999401555955, average_step: 22.84046692607004
```

* valid scores: np.mean([0.9849094567404426, 0.9919517102615694, 1.0])
* 0.9922870556673372

* test scores: np.mean([0.9251944943147815, 0.9539198084979055, 0.9230999401555955])
* 0.9340714143227608

* 测试性能倒也差不了多少就是了