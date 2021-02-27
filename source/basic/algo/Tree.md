## 基本知识
-----
### 目录
* [二叉树](#二叉树)
    * [二叉树的性质](#二叉树的性质)
    * [满二叉树](#满二叉树)
    * [完全二叉树](#完全二叉树)
    * [树类搜索算法](#树类搜索算法)
* [二叉树的存储结构](#二叉树的存储结构)
* [堆](#堆)
* [霍夫曼树](#霍夫曼树)
* [二叉排序树](#二叉排序树)
* [平衡二叉树](#平衡二叉树)
* [B-树](#B-树)
* [Trie 树 - 字典树](#Trie%20树%20-%20字典树)
* [字典树-待补充](#)
* [红黑树-待补充](#)
* [并查集](#并查集)
* [二叉树基本计算公式](#二叉树基本计算公式)
* [LeetCode 例题](#例题)

-----------------
### 二叉树
[参考](https://www.cnblogs.com/A-FM/p/11604720.html)


![Tree 1](img/tree_1.png)
![Tree 2](img/tree_2.png)

**二叉树**
二叉树是有限个结点的集合，这个集合或者是空集，或者是由一个根结点和两株互不相交的二叉树组成，其中一株叫根的做左子树，另一棵叫做根的右子树。

**二叉树的性质**

![Tree 3](img/tree_3.png)

**满二叉树**
深度为k且有2^k －1个结点的二叉树称为满二叉树

![Tree 4](img/tree_4.png)

**完全二叉树**
深度为 k 的，有n个结点的二叉树，当且仅当其每个结点都与深度为 k 的满二叉树中编号从 1 至 n 的结点一一对应，称之为完全二叉树。（除最后一层外，每一层上的节点数均达到最大值；在最后一层上只缺少右边的若干结点）

![Tree 5](img/tree_5.png)

* 性质4：具有 n 个结点的完全二叉树的深度为 log2n + 1

**注意**

* 仅有前序和后序遍历，不能确定一个二叉树，必须有中序遍历的结果


**树的遍历**

* 先序遍历
* 中序遍历
* 后序遍历
* 层次遍历 (使用队列 queue 完成)

**树类搜索算法**

[参考](https://www.acwing.com/blog/content/173/)
* 一般来说就是深度优先搜索,广度优先搜索,A搜索,IDA搜索等几种，通常用的最多就是DFS和BFS。

### 二叉树的存储结构

![Tree 6](img/tree_6.png)
![Tree 7](img/tree_7.png)
![Tree 8](img/tree_8.png)

-------

### 堆

如果一棵完全二叉树的任意一个非终端结点的元素都不小于其左儿子结点和右儿子结点（如果有的话）
的元素，则称此完全二叉树为最大堆。

同样，如果一棵完全二叉树的任意一个非终端结点的元素都不大于其左儿子结点和右儿子结点（如果
有的话）的元素，则称此完全二叉树为最小堆。

**最大堆的根结点中的元素在整个堆中是最大的；**

**最小堆的根结点中的元素在整个堆中是最小的。**


-----

### 霍夫曼树

* 定义：给定n个权值作为n的叶子结点，构造一棵二叉树，若带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman tree)。

* 构造：

    假设有n个权值，则构造出的哈夫曼树有n个叶子结点。 n个权值分别设为 w1、w2、…、wn，则哈夫曼树的构造规则为：

    1. 将w1、w2、…，wn看成是有 n 棵树的森林(每棵树仅有一个结点)；
    2. 在森林中选出两个根结点的权值最小的树合并，作为一棵新树的左、右子树，且新树的根结点权值为其左、右子树根结点权值之和；
    3. 从森林中删除选取的两棵树，并将新树加入森林；
    4. 重复(2)、(3)步，直到森林中只剩一棵树为止，该树即为所求得的哈夫曼树。

------

### 二叉排序树

二叉排序树（Binary Sort Tree）又称二叉查找树（Binary Search Tree），亦称二叉搜索树。

二叉排序树或者是一棵空树，或者是具有下列性质的二叉树：

1. 若左子树不空，则左子树上所有结点的值均小于它的根结点的值；
2. 若右子树不空，则右子树上所有结点的值均大于或等于它的根结点的值；
3. 左、右子树也分别为二叉排序树；
4. 没有键值相等的节点

二分查找的时间复杂度是O(log(n))，最坏情况下的时间复杂度是O(n)（相当于顺序查找）

```cpp
struct Tree{
    int val;
    Tree *left;
    Tree *right;
    Tree():val(0),left(nullptr),right(nullptr) {};
    Tree(int x):val(x),left(nullptr),right(nullptr) {};
    Tree(int x, Tree *&left, Tree *&right):val(x),right(right),left(left) {};
};

Tree *build_insert(Tree *&root, int val){
    if (root == NULL){
        root = new Tree(val);
        return root;
    }
    if (val < root->val){
        root->left = build_insert(root->left,val);
    }else{
        root->right = build_insert(root->right,val);
    }
    return root;
}
```


-----

### 平衡二叉树

平衡二叉树（balanced  binary  tree）,又称 AVL 树。它或者是一棵空树,或者是具有如下性质的二叉树：

1. 它的左子树和右子树都是平衡二叉树，
2. 左子树和右子树的深度之差的绝对值不超过1。

平衡二叉树是在二叉排序树的插入和删除之后加树的旋转，旋转算法是难点。实现时需加入平衡因子。

* 核心思想： 让同一个root的两个子树的高度之差的绝对值不超过1，因此，我们需要进行“换根”操作，即旋转，旋转又分单旋转和双旋转。

平衡二叉树是对二叉搜索树(又称为二叉排序树)的一种改进。二叉搜索树有一个缺点就是，树的结构是无法预料的，随意性很大，它只与节点的值和插入的顺序有关系，往往得到的是一个不平衡的二叉树。在最坏的情况下，可能得到的是一个单支二叉树，其高度和节点数相同，相当于一个单链表，对其正常的时间复杂度有O(log(n))变成了O(n)，从而丧失了二叉排序树的一些应该有的优点。

![Tree 28](img/tree_28.png)
![Tree 29](img/tree_29.png)
![Tree 30](img/tree_30.png)
![Tree 31](img/tree_31.png)
![Tree 32](img/tree_32.png)

* 旋转方式解析

![Tree 33](img/tree_33.png)
![Tree 34](img/tree_34.png)
![Tree 35](img/tree_35.png)
![Tree 36](img/tree_36.png)
![Tree 37](img/tree_37.png)
![Tree 38](img/tree_38.png)
![Tree 39](img/tree_39.png)
![Tree 40](img/tree_40.png)
![Tree 41](img/tree_41.png)


--------

### B-树

**B-树**：B-树是一种非二叉的查找树， 除了要满足查找树的特性，还要满足以下结构特性：

一棵 m 阶的B-树：

1. 树的根或者是一片叶子(一个节点的树),或者其儿子数在 2 和 m 之间。
2. 除根外，所有的非叶子结点的孩子数在 m/2 和 m 之间。
3. 所有的叶子结点都在相同的深度。

B-树的平均深度为logm/2(N)。执行查找的平均时间为O(logm)；

-----

### Trie 树 - 字典树

Trie 树，又称前缀树，字典树， 是一种有序树，用于保存关联数组，其中的键通常是字符串。与二叉查找树不同，键不是直接保存在节点中，而是由节点在树中的位置决定。一个节点的所有子孙都有相同的前缀，也就是这个节点对应的字符串，而根节点对应空字符串。一般情况下，不是所有的节点都有对应的值，只有叶子节点和部分内部节点所对应的键才有相关的值。

Trie 树查询和插入时间复杂度都是 O(n)，是一种以空间换时间的方法。当节点树较多的时候，Trie 树占用的内存会很大。

Trie 树常用于搜索提示。如当输入一个网址，可以自动搜索出可能的选择。当没有完全匹配的搜索结果，可以返回前缀最相似的可能。

------

### 并查集

[参考1](https://segmentfault.com/a/1190000022952886?utm_source=sf-related)
[参考2](https://segmentfault.com/a/1190000004023326)
[参考3](https://zhuanlan.zhihu.com/p/93647900/)

![Tree 21](img/tree_21.png)
![Tree 11](img/tree_11.png)
![Tree 12](img/tree_12.png)
![Tree 13](img/tree_13.png
![Tree 14](img/tree_14.png)
![Tree 15](img/tree_15.png)
![Tree 22](img/tree_22.png)
![Tree 23](img/tree_23.png)
![Tree 24](img/tree_24.png)
![Tree 25](img/tree_25.png)
![Tree 26](img/tree_26.png)
![Tree 27](img/tree_27.png)
![Tree 16](img/tree_16.png)
![Tree 17](img/tree_17.png)
![Tree 18](img/tree_18.png)
![Tree 19](img/tree_19.png)
![Tree 20](img/tree_20.png)


-----
## 二叉树基本计算公式

![Tree 9](img/tree_9.png)
![Tree 10](img/tree_10.png)

------

## 例题

### 二叉树的遍历

#### 二叉树前中后序遍历

二叉树的前中后序遍历，使用递归算法实现最为简单，前序遍历（[LeetCode 144](https://leetcode.com/problems/binary-tree-preorder-traversal/)）为例：

```cpp
struct Tree{
    int val;
    Tree *left;
    Tree *right;
    Tree():val(0),left(nullptr),right(nullptr) {};
    Tree(int x):val(x),left(nullptr),right(nullptr) {};
    Tree(int x, Tree *&left, Tree *&right):val(x),right(right),left(left) {};
};

// 先序遍历 递归
void search_xian(Tree *&root){
    Tree *t = root;
    if (root != NULL){
        cout << root->val << endl;
        search_xian(root->left);
        search_xian(root->right);
    }
    root = t;
}

// 中序遍历 递归
void search_zhong(Tree *&root){
    Tree *t = root;
    if (root != NULL){
        search_zhong(root->left);
        cout << root->val << endl;
        search_zhong(root->right);
    }
    root = t;
}

// 后序遍历 递归
void search_hou(Tree *&root){
    Tree *t = root;
    if (root != NULL){
        search_hou(root->left);

        search_hou(root->right);
        cout << root->val << endl;
    }
    root = t;
}
```

二叉树的非递归遍历，主要的思想是使用栈（Stack）来进行存储操作，记录经过的节点。

非递归前序遍历（[LeetCode 144](https://leetcode.com/problems/binary-tree-preorder-traversal/)）：
非递归不是我写的，仅做参考
```cpp
vector<int> preorderTraversal(TreeNode* root) {
    TreeNode *p = root;
    vector<int> result;
    if (!p) {
        return result;
    }

    stack<TreeNode *> q;
    while (p || !q.empty()) {
        if (p) {
            result.push_back(p->val);
            q.push(p);
            p = p->left;
        }
        else {
            p = q.top();
            q.pop();
            p = p->right;
        }
    }
    return result;
}
```

非递归中序遍历（[LeetCode 94](https://leetcode.com/problems/binary-tree-inorder-traversal/)）：

```cpp
vector<int> inorderTraversal(TreeNode* root) {
    TreeNode *p = root;
    vector<int> result;
    if (!p) {
        return result;
    }

    stack<TreeNode *> q;
    while (p || !q.empty()) {
        if (p) {
            q.push(p);
            p = p->left;
        }
        else {
            p = q.top();
            result.push_back(p->val);
            q.pop();
            p = p->right;
        }
    }
    return result;
}
```

非递归遍历中，后序遍历相对更难实现，因为需要在遍历完左右子节点之后，再遍历根节点，因此不能直接将根节点出栈。这里使用一个 last 指针记录上次出栈的节点，当且仅当节点的右孩子为空（top->right == NULL），或者右孩子已经出栈（top->right == last），才将本节点出栈：

非递归后序遍历（[LeetCode 145](https://leetcode.com/problems/binary-tree-postorder-traversal/)）：

```cpp
 vector<int> postorderTraversal(TreeNode* root) {
    TreeNode *p = root;
    vector<int> result;
    if (!p) {
        return result;
    }

    TreeNode *top, *last = NULL;
    stack<TreeNode *> q;
    while (p || !q.empty()) {
        if (p) {
            q.push(p);
            p = p->left;
        } else {
            top = q.top();
            if (top->right == NULL || top->right == last) {
                q.pop();
                result.push_back(top->val);
                last = top;
            } else {
                p = top->right;
            }
        }
    }

    return result;
}
```

#### 二叉树层序遍历 [LeetCode 102](https://leetcode.com/problems/binary-tree-level-order-traversal/)

二叉树层序遍历有两种方法，分别是深度优先和广度优先：

深度优先（DFS）实现：

```cpp
void traversal(TreeNode *root, int level, vector<vector<int>> &result) {
    if (!root) {
        return;
    }
    // 保证每一层只有一个vector
    if (level > result.size()) {
        result.push_back(vector<int>());
    }
    result[level-1].push_back(root->val);
    traversal(root->left, level+1, result);
    traversal(root->right, level+1, result);
}

vector<vector<int> > levelOrder(TreeNode *root) {
    vector<vector<int>> result;
    traversal(root, 1, result);
    return result;
}
```

广度优先（BFS）实现：

```cpp
struct Tree{
    int val;
    Tree *left;
    Tree *right;
    Tree():val(0),left(nullptr),right(nullptr) {};
    Tree(int x):val(x),left(nullptr),right(nullptr) {};
    Tree(int x, Tree *&left, Tree *&right):val(x),right(right),left(left) {};
};

// 层次遍历 BFS 广度优先
void search_ceng(Tree *&root){
    Tree *t = root;
    queue<Tree*> q;
    while (root!= NULL){
        cout << root->val << endl;
        if (root->left) {q.push(root->left);}
        if (root->right) {q.push(root->right);}
        if (q.empty()) { break; }
        root = q.front();
        q.pop();
    }
    root = t;
}
```

### 二叉树子树 [LeetCode 572](https://leetcode.com/problems/subtree-of-another-tree/)

判断二叉树是否是另一棵二叉树的子树，这里的子树指结构完全相同，所以需要注意叶子节点的指针也需相同，使用递归实现：

```cpp
bool isSubtree(TreeNode* s, TreeNode* t) {
    if (!s) return false;
    if (sameTree(s, t)) return true;
    return isSubtree(s->left, t) || isSubtree(s->right, t);
}

bool sameTree(TreeNode* s, TreeNode* t) {
    if (!s && !t) return true;
    if (!s || !t) return false;
    if (s->val != t->val) return false;
    return sameTree(s->left, t->left) && sameTree(s->right, t->right);
}
```

### 翻转二叉树 [LeetCode 226](https://leetcode.com/problems/invert-binary-tree/)

交互树的左右儿子节点，使用递归实现：

```cpp

Tree *inverse(Tree *&tree){
    if (tree == NULL){
        return tree;
    }
    Tree *tmp = tree->left;
    tree->left = tree->right;
    tree->right = tmp;
    inverse(tree->left);
    inverse(tree->right);
    return tree;
}
```
#### 由 先序遍历 和 中序遍历 重建二叉树 

```cpp

// 由 先序遍历 和 中序遍历 重建二叉树

Tree* rebuild(vector<int>& preorder,vector<int>& inorder){
    if(preorder.size()==0||inorder.size()==0)return NULL;

    Tree* root= new Tree(preorder[0]);//创建当前的根节点
    int i=0;
    while(inorder[i]!=preorder[0])i++;//找到当前根节点在中序遍历中的位置i
    int left=i;   //左子树的长度
    int right=inorder.size()-i -1;  //右子树的长度
    vector<int> sub_left_preorder,sub_left_inorder,sub_right_preorder,sub_right_inorder={};
    for (int j = 0; j < i; ++j) {
        sub_left_inorder.push_back(inorder[j]);
    }
    for (int j = sub_left_inorder.size()+1; j < inorder.size(); ++j) {
        sub_right_inorder.push_back(inorder[j]);
    }

    for (int j = 0; j < preorder.size(); ++j) {
        for (int k = 0; k < sub_left_inorder.size(); ++k) {
            if (preorder[j] == sub_left_inorder[k]){
                sub_left_preorder.push_back(preorder[j]);
            }
        }
        for (int k = 0; k < sub_right_inorder.size(); ++k) {
            if (preorder[j] == sub_right_inorder[k]){
                sub_right_preorder.push_back(preorder[j]);
            }
        }
    }
    if(left>0)root->left=rebuild(sub_left_preorder,sub_left_inorder);
    if(right>0)root->right=rebuild(sub_right_preorder,sub_right_inorder);
    return root;
}

或者 单独记录左右子树序列位置，进行递归判断

Tree* build(vector<int>& preorder,int a1,int b1,vector<int>& inorder,int a2,int b2){
    Tree* root= new Tree(preorder[a1]);//创建当前的根节点
    int i=a2;
    while(inorder[i]!=preorder[a1])i++;//找到当前根节点在中序遍历中的位置i
    int left=i-a2;   //左子树的长度
    int right=b2-i;  //右子树的长度
    if(left>0)root->left=build(preorder,a1+1,a1+left,inorder,a2,i-1);
    if(right>0)root->right=build(preorder,a1+left+1,b1,inorder,i+1,b2);
    return root;
}

Tree* buildTree(vector<int>& preorder, vector<int>& inorder) {
    //二叉树的前序遍历中的第一位一定是根节点
    if(preorder.size()==0||inorder.size()==0)return NULL;
    //找到根节点在中序遍历中的位置，中序遍历之前的节点都是左子树节点，之后都是右子树节点
    return build(preorder,0,preorder.size()-1,inorder,0,inorder.size()-1);
}

```



#### 判断树是否是平衡二叉树 [LeetCode 110](https://leetcode.com/problems/balanced-binary-tree/)
```cpp

int get_depth(Tree *&tree){
    if (tree == NULL){
        return 0;
    }
    int left = get_depth(tree->left);
    int right = get_depth(tree->right);
    return 1+ max(left,right);
}

// 判断是否是平衡二叉树
bool is_balanced_tree(Tree *&tree) {
    if (tree == NULL) return true;
    int left = get_depth(tree->left);
    int right = get_depth(tree->right);
    if (abs(left-right) >1){
        return false;
    }
    return is_balanced_tree(tree->left) && is_balanced_tree(tree->right);
}

```



### 参考资料

* [百度百科：哈弗曼树](http://baike.baidu.com/view/127820.htm)
* [百度百科：二叉排序树](http://baike.baidu.com/view/647462.htm)
* [百度百科：平衡二叉树](http://baike.baidu.com/view/593144.htm)
* [平衡二叉树及其应用场景](http://blog.csdn.net/huiguixian/article/details/6360682)
* [百度百科：B-树](http://baike.baidu.com/view/2228473.htm)
* [前缀树](http://www.iteye.com/topic/1132573)
* [百度百科：前缀树](http://baike.baidu.com/view/9875057.htm)