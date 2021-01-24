## 链表

-------
### C++ 
涉及以下操作
1. 链表定义
2. 插入函数
3. 在指定位置插入
4. 删除指定数据节点
5. 修改指定数据
6. 反转链表 (递归)
7. 打印
8. 获取链表节点个数
8. 待补充
    * 检测是否有环,找到环的入口
    * 循环链表
    * 逆序遍历
    * 单链表排序
    * [在 O(1) 时间删除链表节点](https://zhuanlan.zhihu.com/p/43142694)
        * [删除单链表倒数第 n 个节点](https://zhuanlan.zhihu.com/p/43142694)
        * 判断两个无环单链表是否相交
        * 两个链表相交扩展：求两个无环单链表的第一个相交点
        * 两个链表相交扩展：判断两个有环单链表是否相交

```cpp

// 在链表的后面插入
// 前插
// 在指定位置后插入
// 查看链表长度
// 打印链表
// 删除指定位置元素
// 修改指定位置数据
// 反转链表-递归 LeetCode 206
// 反转链表-非递归



#include <iostream>
using namespace std;

struct ListNode{
    float data;
    ListNode *next;
};

// 在链表的后面插入
// 指针的引用 使用 *&
void insert(struct ListNode *&Node, float num){
    ListNode *per = Node;
    // 这地方的关键是，插入前要把 Node 指针移动到链表的最后，否则就是不停地在第二个节点多次赋值
    while (per->next != NULL){
        per = per->next;
    }
    ListNode *NewNode = new ListNode;
    NewNode->data = num;
    NewNode->next = NULL;
    per->next = NewNode;
}

// 前插
void insert_forward(ListNode *&Node, float num){
    ListNode *temp = new ListNode;
    temp->data = num;
    temp->next = Node;
    Node = temp;
}

// 在指定位置后插入
void insert_in_loc(ListNode *&Node, int loc, float num){
    ListNode *per = Node;
    for (int i = 0; i < loc - 1 ; ++i) {
        Node = Node->next;
    }
    ListNode *temp = new ListNode;
    temp->data = num;
    temp->next = Node->next;
    Node->next = temp;

    Node = per;

}

// 查看链表长度
int check_length(ListNode *&Node){
    // 每一次遍历都需要使用临时指针，遍历完再将头指针给到原来的链表
    ListNode *per = Node;
    int i=0;
    if (Node != NULL) {
        do {
            ++i;
            // cout << Node->data << endl;
            Node = Node->next;
        }while (Node != NULL);
    }
    Node = per;
    return i;
}

// 打印链表
void show_list(ListNode *&Node){
    ListNode *per = Node;
    cout << "show_list: " << endl;
    if (Node != NULL) {
        do {
            cout << Node->data << endl;
            Node = Node->next;
        } while (Node != NULL);
    }
    Node = per;
    cout << "show_list end " << endl;
}

// 删除指定位置元素
void delete_loc_node(ListNode *&Node, int loc){
    ListNode *per = Node;
    int length = check_length(Node);
    if (loc == 0){
        Node = Node->next;
    } else {
        for (int i = 0; i < loc - 2; ++i) {
            Node = Node->next;
        }
        ListNode *temp = Node->next->next;
        Node->next = temp;
        Node = per;
    }
}

// 修改指定位置数据
void alert_loc_data(ListNode *&Node, int loc,float num){
    ListNode *per = Node;
    if (loc == 0){
        Node->data = num;
    }else {
        for (int i = 0; i < loc - 1; ++i) {
            Node = Node->next;
        }
        Node->data = num;
        Node=per;
    }
}

// 反转链表-递归
ListNode* reverseList_recursion(ListNode *&head)
{
    if (head == NULL || head->next == NULL) return head;
    ListNode *p = reverseList_recursion(head->next);
    head->next->next = head;
    head->next = NULL;
    return p;
}

// 反转链表-非递归
ListNode* reverseList(ListNode* head) {
    ListNode* cur = NULL, *pre = head;
    while (pre != NULL) {
        ListNode* t = pre->next;
        pre->next = cur;
        cur = pre;
        pre = t;
    }
    return cur;
}

// 融合两个排序链表
ListNode* mergeTwoLists(ListNode *&l1, ListNode *&l2) {
    if(l1 == NULL)
        return l2;
    if(l2 == NULL)
        return l1;
    ListNode* merge = NULL;
    if(l1->val<=l2->val){
        merge = l1;
        merge->next = mergeTwoLists(l1->next,l2);
    }
    else{
        merge = l2;
        merge->next = mergeTwoLists(l1,l2->next);
    }
    return merge;
}



int main() {
    struct ListNode *Node = nullptr;
    Node = new ListNode;
    Node->data = 0;
    Node->next = NULL;
    insert_forward(Node, -1);
    insert(Node, 1);
    insert(Node, 2);
    insert(Node, 3);
    insert_in_loc(Node,3,1.5);
    delete_loc_node(Node,4);
    //show_list(Node);
    alert_loc_data(Node,5,5);
    show_list(Node);
    int length = check_length(Node);
    cout << "length " << length << endl;
    cout << " " << endl;

    ListNode *new_node = reverseList(Node);
    show_list(new_node);

}


```



-----------
## 例题

#### 单链表翻转 [LeetCode 206](https://leetcode.com/problems/reverse-linked-list/)

```cpp
// 反转链表-递归
ListNode* reverseList_recursion(ListNode *&head)
{
    if (head == NULL || head->next == NULL) return head;
    ListNode *p = reverseList_recursion(head->next);
    head->next->next = head;
    head->next = NULL;
    return p;
}

// 反转链表-非递归
ListNode* reverseList(ListNode* head) {
    ListNode* cur = NULL, *pre = head;
    while (pre != NULL) {
        ListNode* t = pre->next;
        pre->next = cur;
        cur = pre;
        pre = t;
    }
    return cur;
}
```


#### 单链表判断是否有环 [LeetCode 141](https://leetcode.com/problems/linked-list-cycle/)

最容易想到的思路是存一个所有 Node 地址的 Hash 表，从头开始遍历，将 Node 存到 Hash 表中，如果出现了重复，则说明链表有环。

一个经典的方法是双指针（也叫快慢指针），使用两个指针遍历链表，一个指针一次走一步，另一个一次走两步，如果链表有环，两个指针必然相遇。

双指针算法实现：

```cpp
bool hasCycle(ListNode *head) {
    if (head == nullptr) {
        return false;
    }
    ListNode *fast,*slow;
    slow = head;
    fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            return true;
        }
    }
    return false;
}
```

#### 单链表找环入口 [LeetCode 141](https://leetcode.com/problems/linked-list-cycle-ii/)

作为上一题的扩展，为了找到环所在的位置，在快慢指针相遇的时候，此时慢指针没有遍历完链表，再设置一个指针从链表头部开始遍历，这两个指针相遇的点，就是链表环的入口。

算法实现：

```cpp
ListNode *detectCycle(ListNode *head) {
    if (head == nullptr) {
        return nullptr;
    }
    ListNode *fast,*slow;
    slow = head;
    fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            ListNode *slow2 = head;
            while (slow2 != slow) {
                slow = slow->next;
                slow2 = slow2->next;
            }
            return slow2;
        }
    }
    return nullptr;
}
```

#### 单链表找交点 [LeetCode 160](https://leetcode.com/problems/intersection-of-two-linked-lists/)

和找环的方法类似，同样可以使用 Hash 表存储所有节点，发现重复的节点即交点。

一个容易想到的方法是，先得到两个链表的长度，然后得到长度的差值 distance，两个指针分别从两个链表头部遍历，其中较长链表指针先走 distance 步，然后同时向后走，当两个指针相遇的时候，即链表的交点：

```cpp
int getListLength(ListNode *head) {
    if (head == nullptr) {
        return 0;
    }
    int length = 0;
    ListNode *p = head;
    while (p!=nullptr) {
        p = p->next;
        length ++;
    }
    return length;
}

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    int lengthA = getListLength(headA);
    int lengthB = getListLength(headB);

    if (lengthA > lengthB) {
        std::swap(headA, headB);
    };
    int distance = abs(lengthB - lengthA);
    ListNode *p1 = headA;
    ListNode *p2 = headB;
    while(distance--) {
        p2 = p2->next;
    }
    while (p1 != nullptr && p2 != nullptr) {
        if (p1 == p2)
            return p1;
        p1 = p1->next;
        p2 = p2->next;
    }
    return NULL;
}
```

另一个较快的方法时，两个指针 pa，pb 分别从 headA，headB开始遍历，当 pa 遍历到尾部的时候，指向 headB，当 pb 遍历到尾部的时候，转向 headA。当两个指针再次相遇的时候，如果两个链表有交点，则指向交点，如果没有则指向 NULL：

```cpp
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode *pa = headA;
    ListNode *pb = headB;

    while (pa != pb) {
        pa = pa != nullptr ? pa->next : headB;
        pb = pb != nullptr ? pb->next : headA;
    }

    return pa;
}
```


#### 单链表找中间节点 [LeetCode 876](https://leetcode.com/problems/middle-of-the-linked-list/)

用快慢指针法,快指针走两个，慢指针走一个，当快指针走到链表结尾时，慢指针刚好走到链表的中间：

```cpp
ListNode* middleNode(ListNode* head) {
    ListNode *slow = head;
    ListNode *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }

    return slow;
}
```

#### 单链表合并 [LeetCode 21](https://leetcode.com/problems/merge-two-sorted-lists/)

两个链表本身都是排序过的，把两个链表从头节点开始，逐个节点开始进行比较，最后剩下的节点接到尾部：

```cpp
// 融合两个排序链表
ListNode* mergeTwoLists(ListNode *&l1, ListNode *&l2) {
    if(l1 == NULL)
        return l2;
    if(l2 == NULL)
        return l1;
    ListNode* merge = NULL;
    if(l1->val<=l2->val){
        merge = l1;
        merge->next = mergeTwoLists(l1->next,l2);
    }
    else{
        merge = l2;
        merge->next = mergeTwoLists(l1,l2->next);
    }
    return merge;
}

```