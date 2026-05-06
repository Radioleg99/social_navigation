这是一个social navigation的项目
主要目标是： 
让机器人能够就是在多人场景中找到最符合社会规范的道路，并加以行动。

当前的idea是：随着VLM的能力越来越强，VLM应该能够找到场景中不同对象的关系，以及对场景进行地图的建模。如果用得到的场景图以及关系图，将其提供给机器人，机器人是否能找到一条最符合社会规范的路并且执行行动。

可能的做法：
方法1. 通过MPPI等model predict control方法，将显式的社交地图通过llm进行cost map实现，然后将cost map提供给mppi, 让机器人进行行动。
方法2. 一些robot learning 相关方法，当前未知怎么做
方法3. 因为我们的环境尺寸固定不变，可以尝试在LLM后面接一个多层感知机（MLP），直接输出costmap。比如我们在一个10m × 10m × 5m的房间里，每个格子是1m × 1m × 1m，那就有500个格子。MLP输出一个500维的向量，每个元素对应相应格子的代价值。



可能用的数据集：
1. robot navigation相关，但大部分数据集没有人的存在，所以hard to say.



存在的问题：
1. 主题较为简单。


方法1思路：
1. 已存在一大段json文档，其中包含对象间关系（主要是人与人间关系），以及场景中各障碍位置
一开始获得的json地图文档需要经过过滤，过滤出来得到：
json for human: 
{
    objectID: 
    object_State:
    interaction_with: 
    location: 
    bbox_size: ? 
}
只选择可能阻挡机器人的object. 应该同样能够prompt出来。
json for object:
{
    objectID: 
    object_State:
    location: 
    bbox_size: ? 
}


2. 将json文档给 微调过的llm, llm 输出 cost map
两种方式的costmap:
2.1 离散建模的map，给定分辨率以及最小单元，然后将其分为等大等距多个区域
    2.1.1 llm输出array, 非常显式的cost
2.2 连续建模的map, 
    2.2.1 llm输出参数，连续代价函数，需要在设计
    2.2.2 总之应该还是：1. 简单避障  2. 局部优化（引入人的变量）

如果要连续建模？ 应该？ 


3. 微调llm, 蒸馏 让其速度变得很快
4. 将得到的costmap 以及 goal 给到 mppi 控制器
4.1 costmap 应该是 就是加入了隐式的对于场景中人物的理解（就是符合social norm) 
5. 机器人规划路线 （机器人规划路线）






