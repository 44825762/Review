## DFS BFS 深度优先搜索 广度优先搜索






### DFS DFS思路
---
### 这是基本实现，后面还有双向DFS待补充
### 深度优先搜索，使用递归实现

大致算法模板：
```python
    def dfs(a,b):
        if 判断是否达到目标，如果没有进行递归搜索
        dfs():加入搜索方向，根据题意。
        
        
    代码样例如下：
    给定数组a,求和结果K，求是否可以得到K
    n = 5;
    a = [4,2,1,0,7];
    k = 13;

    k2 = 15;


    def dfs(i,sum):
        if i  == n:
            return sum == k;
        if dfs(i+1,sum):
            print(a[i]);
            return True;
        if dfs(i+1,sum+a[i]):
            print(a[i]);
            return True;
        return False;


    if(dfs(0,0)):
        print("yes\n");
    else:
        print("NO\n")
```


### BFS
---
### 思路与DFS想通，但是实现略有不同。
### 关键因素： 

* 记录搜索方向

    例如：x = [0, 1,0，-1] y = [1, 0, -1, 0] 结合之后产生四个方向的坐标变化。
---

* 记录状态的队列： 

    一般使用队列实现，也可以使用数组实现。
    状态记录主要记录访问过的点的坐标。
    记录状态的队列就是搜索的队列以及顺序
---

* 距离记录

    在一般的路径搜索，迷宫题目中需要找到最短的路径，加以修改可以返回迷宫路径
---

* 剩下就是和DFS差不多的条件判断及循环搜索
    
* 代码例子如下：
```python
        N = 10;
        M = 10;
        
        # 0 - 墙壁
        # 1 - 通道
        # 3 - 起点
        # 4 - 终点
        
        A = [
            [0,3,0,0,0,0,0,0,1,0],
            [1,1,1,1,1,1,0,1,1,0],
            [1,0,1,0,0,1,0,0,1,0],
            [1,0,1,1,1,1,1,1,1,1],
            [0,0,1,0,0,1,0,0,0,0],
            [1,1,1,1,0,1,1,1,1,0],
            [1,0,0,0,0,0,0,0,1,0],
            [1,1,1,1,0,1,1,1,1,1],
            [1,0,0,0,0,1,0,0,0,1],
            [1,1,1,1,0,1,1,1,4,0],
        ];
        
        distance=[[9999 for i in range(10)] for j in range(10)]
        s = [0,1];
        g = [9,8];
        
        sx=s[0];
        sy=s[1];
        gx=g[0];
        gy=g[1];
        
        # 移动方向，4个方向
        dx=[1, 0, -1, 0];
        dy=[0, 1, 0, -1];
        
        # 表示状态,标记是否访问过
        state=[];
        
        # 标记初始点为访问过的点
        state.append([sx,sy]);
        
        # 设定距离为0
        distance[sx][sy]=0;
        
        while len(state):
            point = state[0];
            del state[0];
            print("***")
            print(state)
            # 判断是否找到出口
            if A[point[0]][point[1]] == 4:
            #if point[0] == gx & point[1] == gy:
                print("YES");
                break;
        
            for i in range(0,4):
                x_index = point[0] + dx[i];
                y_index = point[1] + dy[i];
        
                # 判断移动条件以及是否访问过
                if 0<= x_index <N and 0<= y_index < M:
                    if A[x_index][y_index] != 0 and distance[x_index][y_index] == 9999:
                        #print([x_index,y_index])
        
                        state.append([x_index,y_index]);
                        distance[x_index][y_index] = distance[point[0]][point[1]] + 1;
        
        print(distance[gx][gy])
        #for i in distance:
        #    print(i)
```


