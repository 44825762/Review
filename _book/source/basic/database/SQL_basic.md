### 部分内容：PL/SQL，存储过程 详见对应页面内容
---
##### []加在字段名上，解决数据库保留字的问题，申明其不是保留字
---
### 内连接
    
关键字：inner join on

语句：select * from a_table a inner join b_table b on a.a_id = b.b_id;

说明：组合两个表中的记录，返回关联字段相符的记录，也就是返回两个表的交集（阴影）部分。

![内连接](https://img-blog.csdn.net/20171209135846780?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGxnMTc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 左连接（左外连接）
    
关键字：left join on / left outer join on

语句：select * from a_table a left join b_table b on a.a_id = b.b_id;

说明：left join 是left outer join的简写，它的全称是左外连接，是外连接中的一种。左(外)连接，左表(a_table)的记录将会全部表示出来，而右表(b_table)只会显示符合搜索条件的记录。右表记录不足的地方均为NULL。

![左连接](https://img-blog.csdn.net/20171209142610819?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGxnMTc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 右连接（右外连接）
    
关键字：right join on / right outer join on

语句：select * from a_table a right outer join b_table b on a.a_id = b.b_id;

说明：right join是right outer join的简写，它的全称是右外连接，是外连接中的一种。与左(外)连接相反，右(外)连接，左表(a_table)只会显示符合搜索条件的记录，而右表(b_table)的记录将会全部表示出来。左表记录不足的地方均为NULL。

![右连接](https://img-blog.csdn.net/20171209144056668?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGxnMTc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 全连接（全外连接）- MySQL目前不支持此种方式，可以用其他方式替代解决。Oracle支持。
    
关键字： full outer join 或者 full join

语句：
```sql
    SELECT * 
    FROM TESTA 
    FULL OUTER JOIN TESTB
    ON TESTA.A=TESTB.A
    
    全外连接的等价写法，对同一表先做左连接，然后右连接
    
    SELECT  TESTA.*,TESTB.*
    FROM TESTA
    LEFT OUTER JOIN TESTB
    ON TESTA.A=TESTB.A
    UNION
    SELECT TESTA.*,TESTB.*
    FROM TESTB
    LEFT OUTER JOIN TESTA
    ON TESTA.A=TESTB.A
```

说明：全外连接是在等值连接的基础上将左表和右表的未匹配数据都加上

---
## SQL 基础
---

1. WHERE

    使用WHERE子句是选定返回数据的条件，可以和表的联合一起使用.
    WHERE有时候和从属运算IN或BETWEEN一起使用：

    ```sql
    SELECT * FROM people WHERE state `IN` ('CA','NY');
    SELECT * FROM inventory WHERE prince `BETWEEN` 50 AND 100; 请注意BETWEEN操作将包括边界值（50和100）
    ```

    WHERE也可以和LIKE表达式一起用：
    在LIKE表达式中，%是一种通配符，表示可能的模糊信息。如果想查找在某一确定的位置上有字符的数据时，可以使用另一个通配符——下划线: 

    ```sql
    SELECT * FROM people WHERE firstname LIKE '_o%' 返回firstname中第二个字符为o的数据,
    类似的还有`STARTING WITH`：STARTING WITH子句附加于WHERE子句上，它的作用与LIKE（%）相似。
    ```

2. ORDER BY

    查询输出的结果按一定的排序规则来显示，ORDER BY可以使用多个字段，在ORDER BY后边的DESC表示用降序排列来代替默认的升序排列。
    
    ```sql
    SELECT * FROM customers ORDER BY consumption;# DESC假如你已经知道了你想要进行排序的列是表中的第一列的话，那么你可以用ORDER BY 1 来代替输入列的名字。
    ```

3. GROUP BY

    SQL无法把正常的列和汇总函数结合在一起，这时就需要<font color=red>GROUP BY</font>子句，它可以对SELECT的结果进行分组后在应用汇总函数。当要求分组结果返回多个数值时不能在SELECT子句中使用除分组列以外的列,这将会导致错误的返回值，但是你可以使用在SELECT中未列出的列进行分组。无论在什么情况下进行分组，SELECT语句中出现的字段只能是在GROUP BY中出现过的才可以。
    
    ```sql
    select [columns] from table_name [where..] group by [columns] [having ...]
    ```

4. HAVING

    Having字句与where子句一样可以进行条件判断的，另外Having 子句通常用来筛选满足条件的组，即在分组之后过滤数据。条件中经常包含聚合函数，使用having 条件过滤出特定的组，也可以使用多个分组标准进行分组。

    ```sql
    SELECT
        goods_type,
        COUNT(goods_type)
    FROM
        tb_goods
    GROUP BY
        goods_type
    HAVING
        avg(price) > 100 //这里使用聚合函数
    ```

5. NION与UNION ALL合并

    在数据库中，union和union all关键字都是将两个结果集合并为一个，但这两者从使用和效率上来说都有所不同。

    union在进行表链接后会筛选掉重复的记录，所以在表链接后会对所产生的结果集进行排序运算，删除重复的记录再返回结果。在SQL运行时先取出两个表的结果，再用排序空间进行排序删除重复的记录，最后返回结果集，如果表数据量大的话可能会导致用磁盘进行排序。

    而union all只是简单的将两个结果合并后就返回。这样，如果返回的两个结果集中有重复的数据，那么返回的结果集就会包含重复的数据了。

    * 使用 union 组合查询的结果集有两个最基本的规则：

        1. 所有查询中的列数和列的顺序必须相同。

        2. 数据类型必须兼容

        ```sql
            select * from test_union1
            union
            select * from test_union2

            select * from test_union1
            union all
            select * from test_union2
        ```

6. INTERSECT相交

    INTERSECT运算符是一个集合运算符，它只返回两个查询或更多查询的交集。

    INTERSECT运算符比较两个查询的结果，并返回由左和右查询输出的不同行记录。

    * 要将INTERSECT运算符用于两个查询，应用以下规则：

        1. 列的顺序和数量必须相同。
        
        2. 相应列的数据类型必须兼容或可转换。

        ![图解intersect](http://www.yiibai.com/uploads/images/201707/1907/403100721_13377.png)

        左侧查询产生一个结果集(1,2,3)，右侧查询返回一个结果集(2,3,4)。
        INTERSECT操作符返回包含(2,3)，也就是两个结果集的相叉的行记录。与UNION运算符不同，INTERSECT运算符返回两个集合之间的交点。
        <font color=red>注意，SQL标准有三个集合运算符，包括UNION，INTERSECT和MINUS。</font>

        **不幸的是，MySQL不支持INTERSECT操作符。 但是我们可以模拟INTERSECT操作符。但可以通过使用内连接和DISTINCT进行模拟，实现同样的效果**

7. MINUS相减

    MINUS运算符，用于从另一个结果集中减去一个结果集。

    要使用MINUS运算符，可以编写单独的SELECT语句并将MINUS运算符放在它们之间。 MINUS运算符返回第一个查询生成的唯一行，但不返回第二个查询生成的唯一行。
    
    下图是MINUS运算符的说明。

    ![MINUS](https://www.yiibai.com/uploads/article/2019/01/28/205724_98352.png)

    为了获得结果集，数据库系统执行两个查询并从第二个查询中减去第一个查询的结果集。要使用MINUS运算符，SELECT子句中的列必须匹配，并且必须具有相同或至少可转换的数据类型。
    
    我们经常在ETL中使用MINUS运算符。 ETL是数据仓库系统中的软件组件。 ETL代表Extract，Transform和Load。 ETL负责将数据从源系统加载到数据仓库系统。
    
    完成加载数据后，可以使用MINUS运算符通过从源系统中的数据中减去目标系统中的数据来确保数据已完全加载.

    ```sql
        SELECT 
            employee_id
        FROM
            employees 
        MINUS 
        SELECT 
            employee_id
        FROM
            dependents
        ORDER BY employee_id;
    ```
    
8. <font color=red>SQL分页（返回前几行)</font>

    记住这两句，再加上order by column_name (desc) 就能应付“消费第二多的客户”、“点击量第5到20名”之类的问题的。

    ```sql
    select * from table_name limit 3,1; # 跳过前3条数据，从数据库中第4条开始查询，取一条数据，即第4条数据
    select * from table_name limit 3 offset 1;# 从数据库中的第2条数据开始查询3条数据，即第2条到第4条
    ```
        
9. <font color=red>条件语句</font>

    MySQL里常用的条件语句是Case。Case语句分为两种：简单Case函数和Case搜索函数。

    - 简单Case函数：

    ```sql
        CASE gender WHEN '0' THEN 'male' WHEN '1' THEN 'female' ELSE 'others' END
    ```

    - Case搜索函数：

    ```sql
        CASE WHEN age < 18 THEN '未成年人' WHEN age < 60 THEN '成年人' ELSE '老年人' END
    ```

    Case语句只返回第一个符合条件的结果，剩下的条件会被自动忽略，比如上例中一个数据的age为16，那么它就在第一个case中被返回，不会进入第二个when中进行判断，因此返回'未成年人'而不是'成年人'。

10. <font color=red>关于空值（ISNULL 和 IS NOT NULL）</font>

    NULL 值的处理方式与其他值不同。
    NULL 用作未知的或不适用的值的占位符。
    注释：无法比较 NULL 和 0；它们是不等价的。
    <font color=red>提示：请始终使用 IS NULL 来查找 NULL 值。<font color=red>
    
        我们如何仅仅选取在 "Address" 列中带有 NULL 值的记录呢？
        我们必须使用 IS NULL 操作符：

    ```sql
        SELECT LastName,FirstName,Address FROM Persons
        WHERE Address IS NULL
    ```
        
        我们如何选取在 "Address" 列中不带有 NULL 值的记录呢？
        我们必须使用 IS NOT NULL 操作符：
            SELECT LastName,FirstName,Address FROM Persons
            WHERE Address IS NOT NULL

11. <font color=red>关于窗口函数 (Online Anallytical Processing)</font>
    
    窗口函数是为了对一组数据进行统计之后返回结果和基础信息，比如姓名，班级。普通的avg()方法等只能返回一行统计数据，不能包含基础信息。<窗口函数>的位置，可以放以下两种函数：
    1） 专用窗口函数，包括后面要讲到的rank, dense_rank, row_number等专用窗口函数。
    2） 聚合函数，如sum. avg, count, max, min等
    因为窗口函数是对where或者group by子句处理后的结果进行操作，所以窗口函数原则上只能写在select子句中。
    <font color=red>一定要注意：在SQL处理中，窗口函数都是最后一步执行，而且仅位于Order by字句之前。</font>

    ```sql
        <窗口函数> over (partition by <用于分组的列名>
            order by <用于排序的列名>)
    ```

    MySQL本身是不支持Window Function的（一般翻译为统计分析函数）,现在大部分的数据库语言都支持window function。
    从SQL Server 2005起，SQL Server开始支持窗口函数 (Window Function)，以及到SQL Server 2012，窗口函数功能增强，目前为止支持以下几种窗口函数：
       * 排序函数 (Ranking Function) -> <font color=red>rank()等</font>；
       
            排序函数中，ROW_NUMBER()较为常用，可用于<font color=red>去重</font>、<font color=red>分页</font>、<font color=red>分组中选择数据</font>，<font color=red>生成数字辅助表</font>等等；
            排序函数在语法上要求OVER子句里必须含ORDER BY，否则语法不通过，对于不想排序的场景可以变通。
            例子：
        ![rank例子1](https://pic3.zhimg.com/80/v2-f8c3b3deb99122d75bb506fdbea81c8d_1440w.jpg)![rank例子2](https://pic3.zhimg.com/80/v2-3285d1d648de9f90864000d58847087a_1440w.jpg) 

            以班级“1”为例，这个班级的成绩“95”排在第1位，这个班级的“83”排在第4位。上面这个结果确实按我们的要求在每个班级内，按成绩排名了。
            
            所得到的的SQL语句如下：

        ```sql
            select *,
                rank() over (partition by 班级  
                # Tip: partition by用来对表分组。在这个例子中，所以我们指定了按“班级”分组（partition by 班级）
                                order by 成绩 desc) as ranking
                # Tip: order by子句的功能是对分组后的结果进行排序，默认是按照升序（asc）排列。在本例中（order by 成绩 desc）是按成绩这一列排序，加了desc关键词表示降序排列。
            from 班级表
        ```

            ![partition 和 order by 区别](https://picb.zhimg.com/80/v2-451c70aa24c68aa7142693fd27c85605_1440w.jpg)
    
        * 聚合函数 (Aggregate Function)  -> <font color=red>over()</font>；
        
            聚合函数 over(partition by 字段）— <font color=red>分区</font>
            聚合函数 over(order by 字段) — `框架字句`
                
                partition by字句的优点是：在同一个select语句中，一个窗口函数的计算独立于按其他列分区的其他窗口函数的计算。
                例如下面的查询，返回每个员工、他的部门、他的部门中的员工数、他的职位以及跟他相同职位的员工数：

            ```sql
                select first_name,department_id,count(*) over (partition by department_id) as dept_cnt,
                job_id,
                count(*) over(partition by job_id) as job_cnt
                from employees
                *order by 2
            ```
        
            `框架字句`：当在窗口函数over字句中使用order by 字句时，就指定了两件事：
            1. 分区中的行如何排序
            2. 在计算中包含哪些行
            3. 通过框架字句,允许定义数据的不同“子窗口”，以便在计算中使用，有很多方式可以指定这样的子窗口。如：
            
                    range between unbounded preceding and current row 指定计算当前行开始、当前行之前的所有值；
                    rows between 1 preceding and current row 指定计算当前行的前一行开始，其范围一直延续到当前行；
                    range between current row and unbounded following 指定计算从当前行开始，包括它后面的所有行；
                    rows between current row and 1 following 指定计算当前行和它后面的一行；

            ```sql
                select department_id,first_name,salary,
                    sum(salary) over (order by hire_date range between unbounded preceding and current row) as run_total1,
                    sum(salary) over(order by hire_date rows between 1 preceding and current row) as run_total2,
                    sum(salary) over(order by hire_date range between current row and unbounded following) as run_total3,
                    sum(salary) over(order by hire_date rows between current row and 1 following) as run_total4
                    from employees
                * where department_id=30
            ```

            最终在显示中，每个 sum 都是一列，列名是 as 后面的字符。
        
        * 分析函数 (Analytic Function) ；
        分析函数是以一定的方法在一个与当前行相关的结果子集中进行计算，也称为窗口函数。
        一般结构为：
        
        ```sql
            Function(arg1 , arg2 ……) over(partition by clause order by clause windowing clause )
            
            Windowing clause : rows | range between start_expr and end_expr
            Start_expr is unbounded preceding | current row | n preceding | n following
            End_expr is unbounded following | current row | n preceding | n following
        ```
        
            function：是所调用的接收0个或多个参数的分析函数。分析函数包括<font color=red>Lag</font>、<font color=red>Lead</font>、<font color=red>First_value</font>、<font color=red>Last_value</font>、<font color=red>Rank</font>、<font color=red>Dense_rank</font>、<font color=red>Row_number</font>、<font color=red>Percentile_cont</font>、<font color=red>Ntile</font>、<font color=red>Listagg</font>等。
        
        * NEXT VALUE FOR Function， 这是给sequence专用的一个函数；
    
    [窗口函数参考内容](https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts?hl=zh-cn)
    
12. <font color=red>相关子查询 -> sql嵌套 与 exist</font>
    独立子查询：顾名思义：就是子查询和外层查询不存在任何联系，是独立于外层查询的
    相关子查询：顾名思义：就是子查询里面的条件依赖于外层查询的数据

    1. 高级子查询：
        业务要求： 查询出 order 表面的orderid 以及其 对应的 相邻的前面的和相邻的后面的 orderid（注意由于是订单表，可能前后的订单之间的大小并不是相差1）：使用相关子查询：
        
             select orderid, 
            (
            select MAX(orderid)
            from [Sales.Orders] as innerOrder
            where innerOrder.orderid<outerOrder.orderid 
            ) as primerOrderId, 
            ( 
            select MIN(orderid) 
            from [Sales.Orders] as innerOrder
            where innerOrder.orderid > outerOrder.orderid
             ) as lastOrderId
            from [Sales.Orders] as outerOrder

    2. 连续聚合（使用相关子查询）
        业务要求：对orderid实现 累加的结果作为一个查询字段进行输出 
        
            select orderid,
            (
            select SUM(orderid)
            from [Sales.Orders] as innerOrder
            where innerOrder.orderid<=outerOrder.orderid
            ) as totalOrderId
            from [Sales.Orders] as outerOrder

13. <font color=red>随机抽样</font>
    
    数据量较小可以简单实使用：

        SELECT * FROM table WHERE field=x ORDER BY  RAND() LIMIT n；
        套用结构：
        SELECT * FROM `lz_adv` WHERE `status` = 1 ORDER BY RAND() LIMIT 1;
    
    对于百万千万级表，以下可做到高效查询：
    
        SELECT * FROM `tableName` WHERE id >= (SELECT floor(RAND() * ((SELECT MAX(id) FROM`tableName`) - (SELECT MIN(id) FROM `tableName`)) + (SELECT MIN(id) FROM `tableName`))) ORDER BY id LIMIT N

14. 操作数据 - 增删查改

    * 增： INSERT INTO Persons (LastName, Address) VALUES ('Wilson', 'Champs-Elysees‘）
    * 删：DELETE FROM table_name或DELETE * FROM table_name（注意，删除行，并不删除表）；
    * 查：SELECT * FROM Persons WHERE City LIKE 'N%'
    * 改：UPDATE Person SET Address = 'Zhongshan 23', City = 'Nanjing' WHERE LastName = 'Wilson'
    * sum函数：SELECT SUM(OrderPrice) AS OrderTotal FROM Orders；as表示生成的数据的列名是OrderTotal，Sum只返回统计值
    * count函数：SELECT COUNT(Customer) AS CustomerNilsen FROM Orders WHERE Customer='Carter'；返回指定列的值的数目（NULL 不计入）
        COUNT(*) 函数返回表中的记录数，即表中有多少条记录
        COUNT(DISTINCT column_name) 函数返回指定列的不同值的数目
    * group by:SELECT Customer,OrderDate,SUM(OrderPrice) FROM Orders GROUP BY Customer,OrderDate
    * 多表联合查询：select column1,column2,column3 from table_name1,talbe_name2 where table_name1.column = table_name2.column；
        `注意：多表联合查询两个表中要有一个记录相同信息的列 column。`

15. 创建和操作表

    1. 创建数据库：
        CREATE DATABASE 数据库名;
    2. 连接数据库：
        USE test_sql;
    3. 删除数据库：
        DROP DATABASE test_sql;
    4. 删除表：
        DROP TABLE t_student;
    5. 创建表：
        CREATE TABLE <表名>
              (<列名> <列的数据类型> [<列的约束>]);
              
            CREATE TABLE t_student(
                student_name VARCHAR(10),
                student_birthday DATETIME,
                student_phone INT,
                student_score FLOAT);
    6. 删除表：
        DROP TABLE t_student;
    7. 复制表：
    注意：复制表的同时表的约束并不能复制过来。只复制内容与结构。
        CREATE TABLE copy_student SELECT * FROM t_student; 
    8. 修改表
        * 添加新列：ALTER TABLE t_student ADD student_address VARCHAR(50);
        * 更改列  ：ALTER TABLE t_student CHANGE student_birthday student_age INT;
        * 删除列  ：ALTER TABLE t_student DROP COLUMN student_score;
    9. 数据库完整性
        * 实体完整性--主键约束，唯一约束 (保证一行数据是有效的)： 
        
            `PRIMARY KEY`
            主键列不能为空也不能重复，通常加在表的id列中。
            
            ```sql
                CREATE TABLE t_student(
                    student_id INT PRIMARY KEY,
                    student_name VARCHAR(10),
                    student_birthday DATETIME,
                    student_phone INT,
                    student_score FLOAT);
            ```
            
            `UNIQUE`
            唯一约束是指给定列的值必须唯一，与主键约束不同的是它可以为空。通常加在表中不能重复的信息中，如电话号码。
            
            ```sql
                CREATE TABLE t_student(
                    student_id INT PRIMARY KEY,
                    student_name VARCHAR(10),
                    student_birthday DATETIME,
                    student_phone INT UNIQUE,
                    student_score FLOAT);
            ```
            
        * 域完整性  --非空约束，默认约束 (保证一列数据是有效的)： 
        
            NOT NULL - 非空约束
            非空约束可以加在诸如姓名等列上。
            
            ```sql
                CREATE TABLE t_student(
                    student_id INT PRIMARY KEY,
                    student_name VARCHAR(10) NOT NULL,
                    student_birthday DATETIME,
                    student_phone INT UNIQUE,
                    student_score FLOAT);
            ```
            
            设定默认值后，可以在添加此列时不指定值,数据库会自动填充设定的默认值。 - 默认约束
            DEFAULT
    
            ```sql
                CREATE TABLE t_student(
                    student_id INT PRIMARY KEY,
                    student_name VARCHAR(10) NOT NULL,
                    student_sex VARCHAR(5) DEFAULT '男',
                    student_birthday DATETIME,
                    student_phone INT UNIQUE,
                    student_score FLOAT);
            ```
            
        * 引用完整性--外键约束， (保证引用的编号是有效的)： 
        
            ```sql
                CREATE TABLE t_student(
                    student_id INT PRIMARY KEY,
                    s_c_id INT REFERENCES t_class(class_id),
                    student_name VARCHAR(10) NOT NULL,
                    student_sex VARCHAR(5) DEFAULT '男',
                    student_birthday DATETIME,
                    student_phone INT UNIQUE,
                    student_score FLOAT
                    CONSTRAINT FOREIGN KEY(s_c_id) REFERENCES t_class(class_id); 
            ```
            
        * 用户自定义完整性--主键约束 (保证自定义规则)： 

16. SQL函数
    
    内建 SQL 函数的语法是：
    
    ```sql
        SELECT function(列) FROM 表
    ```
    函数的基本类型是：

    1. Aggregate 函数
    
        Aggregate 函数的操作面向一系列的值，并返回一个单一的值。
        
    2. Scalar 函数
    
        Scalar 函数的操作面向某个单一的值，并返回基于输入值的一个单一的值。

---  
        
| 函数类型 | 函数 | 描述 |
| --- | --- | --- |
| Aggregate functions | AVG(column) | 返回某列的平均值 |
| Aggregate functions | AVG(column) | 返回某列的平均值 |
| Aggregate functions | COUNT(column) | 返回某列的行数 (不包括 NULL 值) |
| Aggregate functions | COUNT(*) | 返回被选行数 |
| Aggregate functions | FIRST(column) | 返回在指定的域中第一个记录的值 |
| Aggregate functions | LAST(column) | 返回在指定的域中最后一个记录的值 |
| Aggregate functions | MAX(column) | 返回某列的最高值 |
| Aggregate functions | MIN(column) | 返回某列的最低值 |
| Scalar 函数 | UCASE() | 将某个域转换为大写 |
| Scalar 函数 | LCASE() | 将某个域转换为小写 |
| Scalar 函数 | MID(c,start[,end]) | 从某个文本域提取字符 |
| Scalar 函数 | LEN() | 返回某个文本域的长度 |
| Scalar 函数 | INSTR(c,char) | 返回在某个文本域中指定字符的数值位置 |
| Scalar 函数 | LEFT(c,number_of_char) | 返回某个被请求的文本域的左侧部分 |
| Scalar 函数 | RIGHT(c,number_of_char) | 返回某个被请求的文本域的右侧部分 |
| Scalar 函数 | ROUND(c,decimals) | 对某个数值域进行指定小数位数的四舍五入 |
| Scalar 函数 | MOD(x,y) | 返回除法操作的余数 |
| Scalar 函数 | NOW() | 返回当前的系统日期 |
| Scalar 函数 | FORMAT(c,format) | 改变某个域的显示方式 |
| Scalar 函数 | DATEDIFF(d,date1,date2) | 用于执行日期计算 |
        
     
