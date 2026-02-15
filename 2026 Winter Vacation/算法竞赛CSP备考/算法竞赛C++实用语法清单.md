## 一、输入输出进阶

### 1.1 cin/cout的高效使用

```cpp
// 加速技巧（必须写在main开头）
ios::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);

// 格式化输出
#include <iomanip>
cout << fixed << setprecision(2) << 3.14159;  // 保留2位小数
cout << setw(5) << setfill('0') << 42;        // 输出"00042"

// 读入一整行（包含空格）
string s;
getline(cin, s);  // 注意：如果之前有cin，需要用cin.ignore()吸收换行

// 读到文件尾
while (cin >> x) { }
while (getline(cin, s)) { }
```

### 1.2 与scanf/printf的对比场景

```cpp
// 何时必须用scanf/printf：
// 1. 读入超大规模数据（>10^6）
// 2. 格式化输入复杂（如"2024-03-15"）
int y, m, d;
scanf("%d-%d-%d", &y, &m, &d);

// 3. 读入字符或字符串时有特殊要求
char c = getchar();  // 读一个字符，包括空格
ungetc(c, stdin);    // 把字符放回缓冲区

// 4. 需要高效率输出
printf("%.2f\n", ans);
```

## 二、引用与指针在竞赛中的选择

### 2.1 引用(&)的使用场景

```cpp
// 1. 函数修改实参（替代指针）
void swap(int &a, int &b) {  // 比用指针更简洁
    int t = a;
    a = b;
    b = t;
}

// 2. 避免拷贝（常用！）
void dfs(vector<int> &path, int depth) {  // 不写&会超时
    // 处理path
    path.push_back(x);
    dfs(path, depth + 1);
    path.pop_back();
}

// 3. 范围for循环修改
for (auto &x : vec) {
    x *= 2;  // 直接修改原元素
}

// 4. 函数返回左值（少见但有用）
int &getElement(vector<int> &v, int i) {
    return v[i];  // 返回引用，可以赋值
}
// getElement(v, 0) = 100;
```

### 2.2 指针在竞赛中的最后用途

```cpp
// 1. 动态数据结构（链表、树）
struct Node {
    int val;
    Node *left, *right;
    Node(int x) : val(x), left(nullptr), right(nullptr) {}
};

// 2. 函数指针（排序自定义）
sort(arr, arr + n, [](int a, int b) { return a > b; });

// 3. 需要空值(nullptr)的场景
if (ptr == nullptr) { }
```

## 三、STL容器详解（竞赛高频）

### 3.1 vector - 动态数组（最常用）

```cpp
#include <vector>

// 初始化
vector<int> v1;                    // 空
vector<int> v2(10);                 // 10个0
vector<int> v3(10, 5);              // 10个5
vector<int> v4 = {1, 2, 3, 4, 5};   // C++11

// 常用操作
v.push_back(6);                     // 尾部插入 O(1)
v.pop_back();                        // 尾部删除 O(1)
v.size();                            // 元素个数
v.empty();                           // 是否为空
v.clear();                           // 清空
v.resize(20);                        // 重新设置大小
v.reserve(1000);                     // 预分配空间（防超时关键！）

// 访问
v[0];                                // 不检查越界
v.at(0);                             // 检查越界，慢一点
v.front();                           // 第一个元素
v.back();                            // 最后一个元素

// 删除指定元素（配合remove）
v.erase(remove(v.begin(), v.end(), val), v.end());  // 删除所有值为val的元素

// 二维vector
vector<vector<int>> matrix(5, vector<int>(3, 0));  // 5行3列全0
```

### 3.2 string - 字符串（比C字符串好用）

```cpp
#include <string>

string s = "hello";
string t = "world";

// 拼接
string res = s + " " + t;           // "hello world"
s += "!!!";                          // "hello!!!"

// 子串
string sub = s.substr(1, 3);         // 从1开始取3个字符："ell"
string sub2 = s.substr(2);           // 从2到结尾："llo!!!"

// 查找
int pos = s.find("ll");              // 返回2，找不到返回string::npos
if (s.find('x') == string::npos) { }

// 替换
s.replace(1, 2, "aaa");              // 从1开始替换2个字符为"aaa"

// 插入/删除
s.insert(2, "123");
s.erase(1, 3);

// 与数字转换
int num = stoi("123");                // string to int
long long ll = stoll("123456789");
double d = stod("3.14");
string s2 = to_string(12345);         // 数字转字符串

// 按字符遍历
for (char c : s) { }
for (char &c : s) { c = toupper(c); } // 转大写
```

### 3.3 map/unordered_map - 映射（字典）

```cpp
#include <map>
#include <unordered_map>

// map 有序（红黑树） O(logn)
// unordered_map 无序（哈希表） O(1)平均，但可能被卡

map<string, int> mp;
unordered_map<int, vector<int>> ump;  // 值可以是任意类型

// 插入
mp["apple"] = 5;                       // 直接赋值
mp.insert({"banana", 3});               // 插入键值对
mp.emplace("orange", 2);                 // C++11高效插入

// 查找
if (mp.count("apple")) { }              // 检查是否存在
auto it = mp.find("apple");
if (it != mp.end()) {
    cout << it->first << " " << it->second;
}

// 遍历
for (auto &p : mp) {
    cout << p.first << ":" << p.second;
}

// 注意事项：
// 1. 访问不存在的键会创建默认值
int x = mp["new_key"];  // 会创建值为0的键值对
// 2. 可以用find避免创建
if (auto it = mp.find("key"); it != mp.end()) { } // C++17
```

### 3.4 set/unordered_set - 集合

```cpp
#include <set>
#include <unordered_set>

set<int> st;
unordered_set<int> ust;

// 操作
st.insert(5);
st.erase(5);
if (st.count(5)) { }  // 判断是否存在
int sz = st.size();

// set有序，可获取最大最小
int minVal = *st.begin();
int maxVal = *st.rbegin();

// 自定义排序
struct cmp {
    bool operator()(const int &a, const int &b) const {
        return a > b;  // 降序
    }
};
set<int, cmp> st2;
```

### 3.5 stack/queue/deque - 栈、队列、双端队列

```cpp
#include <stack>
#include <queue>
#include <deque>

// 栈
stack<int> st;
st.push(1);          // 入栈
int top = st.top();  // 取栈顶
st.pop();            // 出栈（不返回元素）

// 队列
queue<int> q;
q.push(1);           // 入队
int front = q.front(); // 取队首
int back = q.back();   // 取队尾
q.pop();             // 出队

// 双端队列（可在两端操作）
deque<int> dq;
dq.push_front(1);
dq.push_back(2);
dq.pop_front();
dq.pop_back();
int first = dq[0];   // 可随机访问
```

### 3.6 priority_queue - 优先队列（堆）

```cpp
#include <queue>

// 默认大顶堆（最大元素在顶部）
priority_queue<int> pq;
pq.push(3);
pq.push(1);
pq.push(5);
int top = pq.top();  // 5

// 小顶堆
priority_queue<int, vector<int>, greater<int>> pq_min;

// 自定义结构体的堆
struct Node {
    int x, y;
    bool operator<(const Node &other) const {
        return x > other.x;  // 注意：这里重载<但实现的是>，因为priority_queue默认用<
    }
};
priority_queue<Node> pq_node;

// 或使用lambda（C++11）
auto cmp = [](int a, int b) { return a > b; };
priority_queue<int, vector<int>, decltype(cmp)> pq_custom(cmp);
```

### 3.7 pair/tuple - 对组/元组

```cpp
#include <utility>  // pair
#include <tuple>    // tuple

// pair
pair<int, string> p = {1, "hello"};
p = make_pair(2, "world");
cout << p.first << " " << p.second;

// pair比较：先比较first，再second
vector<pair<int, int>> v;
sort(v.begin(), v.end());  // 默认按first升序，first相同按second

// tuple（C++11）
tuple<int, string, double> t = {1, "test", 3.14};
auto [a, b, c] = t;  // C++17结构化绑定
int x = get<0>(t);   // 按索引获取
```

## 四、算法常用函数

### 4.1 algorithm头文件

```cpp
#include <algorithm>

// 排序
sort(v.begin(), v.end());                       // 默认升序
sort(v.begin(), v.end(), greater<int>());        // 降序
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// 二分查找（必须在有序序列上）
if (binary_search(v.begin(), v.end(), 5)) { }
auto it = lower_bound(v.begin(), v.end(), 5);   // 第一个>=5的位置
auto it = upper_bound(v.begin(), v.end(), 5);   // 第一个>5的位置
int pos = lower_bound(v.begin(), v.end(), 5) - v.begin();  // 下标

// 最值
int mx = *max_element(v.begin(), v.end());
int mn = *min_element(v.begin(), v.end());
auto [min_it, max_it] = minmax_element(v.begin(), v.end()); // C++11

// 反转
reverse(v.begin(), v.end());

// 去重（配合sort使用）
sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end());

// 排列
next_permutation(v.begin(), v.end());  // 下一个排列
prev_permutation(v.begin(), v.end());  // 上一个排列

// 填充
fill(v.begin(), v.end(), 0);
fill_n(v.begin(), 5, 1);                // 前5个填充1

// 复制
copy(v.begin(), v.end(), arr);           // 复制到数组
copy_if(v.begin(), v.end(), back_inserter(res), [](int x) { return x > 0; });

// 累加（numeric头文件）
#include <numeric>
int sum = accumulate(v.begin(), v.end(), 0);
int product = accumulate(v.begin(), v.end(), 1, multiplies<int>());
```

### 4.2 常用函数对象

```cpp
#include <functional>

plus<int>();           // x + y
minus<int>();          // x - y
multiplies<int>();     // x * y
divides<int>();        // x / y
modulus<int>();        // x % y
negate<int>();         // -x

equal_to<int>();       // x == y
not_equal_to<int>();   // x != y
greater<int>();        // x > y
less<int>();           // x < y
greater_equal<int>();  // x >= y
less_equal<int>();     // x <= y

logical_and<bool>();   // x && y
logical_or<bool>();    // x || y
logical_not<bool>();   // !x
```

## 五、C++11/14/17新特性（竞赛常用）

### 5.1 auto类型推导

```cpp
auto a = 10;                 // int
auto b = 3.14;               // double
auto c = "hello";            // const char*

vector<int> v = {1, 2, 3};
for (auto it = v.begin(); it != v.end(); ++it) { }
for (auto x : v) { }         // 范围for
```

### 5.2 Lambda表达式

```cpp
// 基本语法 [捕获列表](参数列表)->返回类型{函数体}

// 简单使用
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// 捕获外部变量
int threshold = 5;
auto cnt = count_if(v.begin(), v.end(), [threshold](int x) { 
    return x > threshold; 
});

// 引用捕获
int sum = 0;
for_each(v.begin(), v.end(), [&sum](int x) { sum += x; });

// 可变lambda
vector<int> v2;
generate_n(back_inserter(v2), 10, [n = 0]() mutable { return n++; });
```

### 5.3 结构化绑定（C++17）

```cpp
pair<int, string> p = {1, "hello"};
auto [id, name] = p;  // id=1, name="hello"

map<string, int> mp;
for (auto [key, value] : mp) {
    cout << key << ":" << value;
}

tuple<int, string, double> t = {1, "test", 3.14};
auto [a, b, c] = t;
```

### 5.4 nullptr和constexpr

```cpp
int *p = nullptr;  // 代替NULL

constexpr int MAXN = 1000;  // 编译期常量
int arr[MAXN];              // 可用于数组大小
```

## 六、算法竞赛实用技巧

### 6.1 类型别名

```cpp
using ll = long long;
using pii = pair<int, int>;
using vi = vector<int>;
using vvi = vector<vi>;

// 复杂类型简化
using func = function<bool(int, int)>;
```

### 6.2 常用宏定义（谨慎使用）

```cpp
#define forn(i, n) for (int i = 0; i < int(n); i++)
#define forr(i, a, b) for (int i = int(a); i <= int(b); i++)
#define all(x) (x).begin(), (x).end()
#define sz(x) (int)(x).size()
#define pb push_back
#define mp make_pair
#define fi first
#define se second
```

### 6.3 输入输出重定向（本地调试）

```cpp
#ifdef LOCAL
    freopen("in.txt", "r", stdin);
    freopen("out.txt", "w", stdout);
#endif
```

### 6.4 常见陷阱与优化

```cpp
// 1. vector<bool>是特化版本，不是bool数组，慎用
vector<char> vbool;  // 用char代替

// 2. string的+=比+效率高
string s = "a";
s = s + "b";  // 创建临时对象
s += "b";     // 直接追加

// 3. reserve预分配空间
vector<int> v;
v.reserve(1000000);  // 避免多次扩容

// 4. emplace_back优于push_back
v.push_back(Node(1, 2));    // 构造+移动
v.emplace_back(1, 2);       // 直接构造

// 5. 移动语义（极少用，了解即可）
vector<int> v2 = move(v1);  // v1被清空，v2获得资源
```

## 七、中档题实战示例

### 例题1：拓扑排序（使用vector, queue, 引用）

```cpp
// 问题：给定n个节点m条边的有向图，输出拓扑排序
vector<vector<int>> adj(n);  // 邻接表
vector<int> indeg(n, 0);     // 入度

// 建图
for (auto [u, v] : edges) {  // C++17结构化绑定
    adj[u].push_back(v);
    indeg[v]++;
}

// 拓扑排序
queue<int> q;
for (int i = 0; i < n; i++) {
    if (indeg[i] == 0) q.push(i);
}

vector<int> ans;
while (!q.empty()) {
    int u = q.front(); q.pop();
    ans.push_back(u);
    
    for (int v : adj[u]) {
        if (--indeg[v] == 0) {
            q.push(v);
        }
    }
}
// 用到：vector, queue, 范围for, auto
```

### 例题2：合并区间（使用pair, sort, lambda）

```cpp
// 问题：合并所有重叠的区间
vector<pair<int, int>> intervals = {{1,3}, {2,6}, {8,10}, {15,18}};

// 按左端点排序
sort(intervals.begin(), intervals.end(), 
     [](auto &a, auto &b) { return a.first < b.first; });

vector<pair<int, int>> ans;
for (auto [l, r] : intervals) {
    if (ans.empty() || ans.back().second < l) {
        ans.emplace_back(l, r);
    } else {
        ans.back().second = max(ans.back().second, r);
    }
}
// 用到：pair, sort, lambda, emplace_back, 结构化绑定
```

### 例题3：字母异位词分组（使用unordered_map, string）

```cpp
// 问题：将字符串数组按字母异位词分组
vector<string> strs = {"eat", "tea", "tan", "ate", "nat", "bat"};

unordered_map<string, vector<string>> mp;
for (string &s : strs) {
    string key = s;
    sort(key.begin(), key.end());  // 排序后的字符串作为key
    mp[key].push_back(s);
}

vector<vector<string>> ans;
for (auto &[k, v] : mp) {
    ans.push_back(v);
}
// 用到：unordered_map, string, sort, 引用, 结构化绑定
```

### 例题4：第K大的数（使用priority_queue）

```cpp
// 问题：找出无序数组中第K大的元素
int findKthLargest(vector<int>& nums, int k) {
    // 小顶堆，维护当前最大的k个数
    priority_queue<int, vector<int>, greater<int>> pq;
    
    for (int x : nums) {
        pq.push(x);
        if (pq.size() > k) {
            pq.pop();  // 弹出最小的
        }
    }
    return pq.top();  // 堆顶是第k大的
}
// 用到：priority_queue, 范围for
```

### 例题5：最长连续序列（使用unordered_set）

```cpp
// 问题：找出未排序数组中最长连续序列的长度
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> st(nums.begin(), nums.end());
    int ans = 0;
    
    for (int x : st) {
        // 只从序列起点开始找
        if (!st.count(x - 1)) {
            int len = 1;
            int cur = x;
            while (st.count(cur + 1)) {
                cur++;
                len++;
            }
            ans = max(ans, len);
        }
    }
    return ans;
}
// 用到：unordered_set, 范围for, count
```

## 八、竞赛常用模板框架

```cpp
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using pii = pair<int, int>;
using vi = vector<int>;
using vvi = vector<vi>;

const int INF = 0x3f3f3f3f;
const int MOD = 1e9 + 7;

void solve() {
    int n;
    cin >> n;
    vi a(n);
    for (auto &x : a) cin >> x;
    
    // 解题逻辑
    ll ans = 0;
    
    cout << ans << '\n';
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T = 1;
    cin >> T;  // 注释掉如果只有一组数据
    while (T--) {
        solve();
    }
    
    return 0;
}
```

建议
1. 先熟悉STL基本容器操作
2. 理解引用传参避免拷贝
3. 掌握lambda表达式简化代码
4. 练习时主动使用新特性
