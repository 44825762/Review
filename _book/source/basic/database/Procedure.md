### 存储过程的概念（Stored Procedure）

1. 存储过程（Stored Procedure）是一组为了完成特定功能的SQL语句集。经编译后存储在数据库中。
2. 存储过程是数据库中的一个重要对象，用户通过指定存储过程的名字并给出参数（可以有参数，也可以没有）来执行它。
3. 存储过程是由 流控制 和 SQL语句书写的过程，这个过程经编译和优化后存储在数据库服务器中。
4. 存储过程 可由应用程序通过一个调用来执行，而且允许用户声明变量。
5. 同时，存储过程可以接收和输出参数、返回执行存储过程的状态值，也可以嵌套调用。

### 存储过程的优点
1. 存储过程的使用大大增强了SQL语言的功能和灵活性。

    存储过程可以用流控制语句编写，有很强的灵活性，可以完成复杂的判断和较复杂的运算。

2. 可保证数据的安全性和完整性。

    通过存储过程可以使没有权限的用户在控制之下间接地存取数据库，从而保证数据的安全。

    通过存储过程可以使相关的动作在一起发生，从而可以维护数据库的完整性。（就像事务的原子性：要么事务内的所有SQL语句全部执行成功，要么全部不成功）

3. 在运行存储过程前，数据库已对其进行了语法和句法分析，并给出了优化执行方案。

    这种已经编译好的过程可极大地改善SQL语句的性能。

    由于执行SQL语句的大部分工作已经完成（因为已经提前经过编译），所以存储过程能以极快的速度执行。

4. 可以降低网络的通信量。

    客户端调用存储过程只需要传存储过程名和相关参数即可，与传输SQL语句相比自然数据量少了很多（在远程访问时体现出来）。

5. 存储过程只在创造时进行编译，以后每次执行存储过程都不需再重新编译，而一般SQL语句每执行一次就编译一次,所以使用存储过程可提高数据库执行速度。

6. 当对数据库进行复杂操作时(如对多个表进行Update,Insert,Query,Delete时)，可将此复杂操作用存储过程封装起来与数据库提供的事务处理结合一起使用。

    比如每一步对数据库的操作用一个事务来完成，把这些事务全都放在一个存储过程中。

7. 存储过程可以重复使用,可减少数据库开发人员的工作量。

8. 安全性高,可设定只有某些用户才具有对指定存储过程的使用权

### 存储过程缺点

1. 调试麻烦：但是用 PL/SQL Developer 调试很方便！弥补这个缺点。

2. 移植问题：数据库端代码当然是与数据库相关的。但是如果是做工程型项目，基本不存在移植问题。

3. 重新编译问题：因为后端代码是运行前编译的，如果带有引用关系的对象发生改变时，受影响的存储过程、包将需要重新编译（不过也可以设置成运行时刻自动编译）。

    比如A存储过程调用B存储过程，使用B的返回值作为参数，如果B的参数或返回值发生改变时，会对调用她的A产生影响，此时存储过程就要重新编译，设置成运行时刻自动编译。

4. 维护比较困难：如果在一个程序系统中大量的使用存储过程，到程序交付使用的时候随着用户需求的增加会导致数据结构的变化，接着就是系统的相关问题了，最后如果用户想维护该系统可以说是很难很难、而且代价是空前的，维护起来更麻烦。

### 存储过程的特性

1. <font color=red>存储过程与函数的区别<font color=red>

    * 返回值：函数只能返回一个变量，而存储过程可以返回多个。对于存储过程来说可以返回参数，如记录集，而函数只能返回值或者表对象
    
    * 存储过程一般是作为一个独立的部分来执行（ EXECUTE 语句执行），而函数可以作为查询语句的一个部分来调用（SELECT调用），由于函数可以返回一个表对象，因此它可以在查询语句中位于FROM关键字的后面。 SQL语句中不可用存储过程，而可以使用函数。
    
    * 存储过程实现的功能要复杂一点，而函数的实现的功能针对性比较强，比较单一。

2. <font color=red>存储过程与事务的区别<font color=red>

    * 存储位置：事务在程序中被调用，保存在调用以及实现它的代码中，存储过程可以在数据库客户端直接被调用，经编译后存储在数据库中。
    
    * 运行方式：事务在每次被调用的时候执行其中的SQL语句，存储过程预先经过编译，并不是每次被调用时都会执行一遍其中的SQL语句。
    
    * 事务有严格的一致性和原子性，使用的安全性高，存储过程则没有这些特性，在进行一些复杂的操作时，为了保证操作的准确性，可以在存储过程中调用事务，然后判断事务的执行结果是否成功来确保操作的准确性。

3. 触发器

    * 概念及作用
    
        触发器是一种特殊类型的存储过程，它不同于我们前面介绍过的存储过程。触发器主要是通过事件进行触发而被执行的，而存储过程可以通过存储过程名字而被直接调用。当对某一表进行诸如Update、 Insert、 Delete 这些操作时，SQL Server就会自动执行触发器所定义的SQL 语句，从而确保对数据的处理必须符合由这些SQL 语句所定义的规则。
    
    * 功能
         <font color=red>触发器的主要作用就是其能够实现由主键和外键所不能保证的复杂的参照完整性和数据的一致性。<font color=red>
         
         除此之外，触发器还有其它许多不同的功能：

        1. 强化约束(Enforce restriction)
        
            触发器能够实现比CHECK 语句更为复杂的约束。
        
        2. 跟踪变化(Auditing changes)
        
            触发器可以侦测数据库内的操作，从而不允许数据库中未经许可的指定更新和变化。
        
        3. 级联运行(Cascaded operation)。
        
            触发器可以侦测数据库内的操作，并自动地级联影响整个数据库的各项内容。例如，某个表上的触发器中包含有对另外一个表的数据操作(如删除，更新，插入)而该操作又导致该表上触发器被触发。
        
        4. 存储过程的调用(Stored procedure invocation)。
        
            为了响应数据库更新,触发器可以调用一个或多个存储过程，甚至可以通过外部过程的调用而在DBMS(数据库管理系统)本身之外进行操作。

        由此可见，触发器可以解决高级形式的业务规则或复杂行为限制以及实现定制记录等一些方面的问题。例如，触发器能够找出某一表在数据修改前后状态发生的差异，并根据这种差异执行一定的处理。此外一个表的同一类(Insert、 Update、Delete)的多个触发器能够对同一种数据操作采取多种不同的处理。
        
        总体而言，触发器性能通常比较低。当运行触发器时，系统处理的大部分时间花费在参照其它表的这一处理上，因为这些表既不在内存中也不在数据库设备上，而删除表和插入表总是位于内存中。可见触发器所参照的其它表的位置决定了操作要花费的时间长短。


### 存储过程的语法和参数

```sql
    --------------创建存储过程-----------------
    
    CREATE PROC [ EDURE ] procedure_name [ ; number ]
        [ { @parameter data_type }
            [ VARYING ] [ = default ] [ OUTPUT ]
        ] [ ,...n ]
    
    [ WITH
        { RECOMPILE | ENCRYPTION | RECOMPILE , ENCRYPTION } ]
    
    [ FOR REPLICATION ]
    
    AS sql_statement [ ...n ]
    
    --------------调用存储过程-----------------
    
    EXECUTE Procedure_name '' --存储过程如果有参数，后面加参数格式为：@参数名=value，也可直接为参数值value
    
    --------------删除存储过程-----------------
    
    drop procedure procedure_name    --在存储过程中能调用另外一个存储过程，而不能删除另外一个存储过程
```

* 存储过程例子

    1. 只返回单一记录集的存储过程 
    
        **结果**：相当于运行 select * from UserAccount 这行代码，结果为整个表的数据。
    
        ```sql
            -------------创建名为GetUserAccount的存储过程----------------
            create Procedure GetUserAccount
            as
            select * from UserAccount
            go
            
            -------------执行上面的存储过程----------------
            exec GetUserAccount
        ```
        
    2. 没有输入输出的存储过程 
    
        **结果**：相当于运行 insert into UserAccount (UserName,[PassWord],RegisterTime,RegisterIP) values(9,9,'2013-01-02',9) 这行代码。
    
        ```sql
            -------------创建名为GetUserAccount的存储过程----------------
            
            create Procedure inUserAccount
            as
            insert into UserAccount (UserName,[PassWord],RegisterTime,RegisterIP) values(9,9,'2013-01-02',9)
            go
            
            -------------执行上面的存储过程----------------
            
            exec inUserAccount
        ```
        
    3. 有返回值的存储过程 
    
        **解释**：这里的@@rowcount为执行存储过程影响的行数，执行的结果是不仅插入了一条数据，还返回了一个值即 return value =1  ，这个可以在程序中获取
    
        ```sql
            -------------创建名为GetUserAccount的存储过程----------------
            
            create Procedure inUserAccountRe
            as
            insert into UserAccount (UserName,[PassWord],RegisterTime,RegisterIP) values(10,10,'2013-01-02',10)
            return @@rowcount
            go
            
            -------------执行上面的存储过程----------------
            
            exec inUserAccountRe
        ```
        
    4. 有输入参数和输出参数的存储过程
    
        **解释**：`@UserName`为输入参数，`@UserID`为输出参数。 运行结果为@userID为COOUT（*）即 =1。
    
        ```sql
            -------------创建名为GetUserAccount的存储过程----------------
            
            create Procedure GetUserAccountRe
            @UserName nchar(20),
            @UserID int output
            as
            if(@UserName>5)
            select @UserID=COUNT(*) from UserAccount where UserID>25
            else
            set @UserID=1000
            go
            
            -------------执行上面的存储过程----------------
            
            exec GetUserAccountRe '7',null
        ```
        
    5. 同时具有返回值、输入参数、输出参数的存储过程 
    
        **结果**：@userID为COOUT（*）即 =1，Retun Value=1。
    
        ```sql
            -------------创建名为GetUserAccount的存储过程----------------
            
            create Procedure GetUserAccountRe1
            @UserName nchar(20),
            @UserID int output
            as
            if(@UserName>5)
            select @UserID=COUNT(*) from UserAccount where UserID>25
            else
            set @UserID=1000
            return @@rowcount
            go
            
            -------------执行上面的存储过程----------------
            
            exec GetUserAccountRe1 '7',null
        ```
        
    6. 返回多个记录集的存储过程
    
        **结果**：返回两个结果集，一个为 select * from UserAccount，另一个为 select * from UserAccount where UserID>5 。
    
        ```sql
            -------------创建名为GetUserAccount的存储过程----------------
            
            create Procedure GetUserAccountRe3
            as
            select * from UserAccount
            select * from UserAccount where UserID>5
            go
            
            -------------执行上面的存储过程----------------
            
            exec GetUserAccountRe3
        ```
---



        
        
        
        
        









