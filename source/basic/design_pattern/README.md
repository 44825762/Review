
### 工厂模式
工厂模式（Factory Pattern）是 Java 中最常用的设计模式之一。这种类型的设计模式属于创建型模式，它提供了一种创建对象的最佳方式。
在工厂模式中，我们在创建对象时不会对客户端暴露创建逻辑，并且是通过使用一个共同的接口来指向新创建的对象。
##### 主要解决：主要解决接口选择的问题。
##### 何时使用：我们明确地计划不同条件下创建不同实例时。
##### 优点： 
1. ###### 一个调用者想创建一个对象，只要知道其名称就可以了。
2. ###### 扩展性高，如果想增加一个产品，只要扩展一个工厂类就可以。
3. ###### 屏蔽产品的具体实现，调用者只关心产品的接口。

##### 缺点：每次增加一个产品时，都需要增加一个具体类和对象实现工厂，使得系统中类的个数成倍增加，在一定程度上增加了系统的复杂度，同时也增加了系统具体类的依赖。这并不是什么好事。

![工厂模式](https://www.runoob.com/wp-content/uploads/2014/08/AB6B814A-0B09-4863-93D6-1E22D6B07FF8.jpg)

##### 创建步骤：
1. ##### 步骤 1
    ##### 创建一个接口:

        Shape.java
        public interface Shape {
           void draw();
        }

2. ##### 步骤 2
    ##### 创建实现接口的实体类：

        Rectangle.java
        public class Rectangle implements Shape {
         
           @Override
           public void draw() {
              System.out.println("Inside Rectangle::draw() method.");
           }
        }
        Square.java， Circle.java 同上
        

3. ##### 步骤 3
    ##### 创建一个工厂，生成基于给定信息的实体类的对象:

        ShapeFactory.java
        public class ShapeFactory {
            
           //使用 getShape 方法获取形状类型的对象
           public Shape getShape(String shapeType){
              if(shapeType == null){
                 return null;
              }        
              if(shapeType.equalsIgnoreCase("CIRCLE")){
                 return new Circle();
              } else if(shapeType.equalsIgnoreCase("RECTANGLE")){
                 return new Rectangle();
              } else if(shapeType.equalsIgnoreCase("SQUARE")){
                 return new Square();
              }
              return null;
           }
        }

4. ##### 步骤 4
    ##### 使用该工厂，通过传递类型信息来获取实体类的对象:

        FactoryPatternDemo.java
        public class FactoryPatternDemo {
           public static void main(String[] args) {
              ShapeFactory shapeFactory = new ShapeFactory();
         
              //获取 Circle 的对象，并调用它的 draw 方法
              Shape shape1 = shapeFactory.getShape("CIRCLE");
         
              //调用 Circle 的 draw 方法
              shape1.draw();
         
              //获取 Rectangle 的对象，并调用它的 draw 方法
              Shape shape2 = shapeFactory.getShape("RECTANGLE");
         
              //调用 Rectangle 的 draw 方法
              shape2.draw();
         
              //获取 Square 的对象，并调用它的 draw 方法
              Shape shape3 = shapeFactory.getShape("SQUARE");
         
              //调用 Square 的 draw 方法
              shape3.draw();
           }
        }

5. ##### 步骤 5
    ##### 执行程序，输出结果：
    
        Inside Circle::draw() method.
        Inside Rectangle::draw() method.
        Inside Square::draw() method.

---

----
### 建造者模式
建造者模式（Builder Pattern）使用多个简单的对象一步一步构建成一个复杂的对象。这种类型的设计模式属于创建型模式，它提供了一种创建对象的最佳方式。
一个 Builder 类会一步一步构造最终的对象。该 Builder 类是独立于其他对象的。
##### 意图：将一个复杂的构建与其表示相分离，使得同样的构建过程可以创建不同的表示。

##### 主要解决：主要解决在软件系统中，有时候面临着"一个复杂对象"的创建工作，其通常由各个部分的子对象用一定的算法构成；由于需求的变化，这个复杂对象的各个部分经常面临着剧烈的变化，但是将它们组合在一起的算法却相对稳定。
##### 何时使用：一些基本部件不会变，而其组合经常变化的时候。
##### 优点： 1、建造者独立，易扩展。 2、便于控制细节风险。

##### 缺点：1、产品必须有共同点，范围有限制。 2、如内部变化复杂，会有很多的建造类。

![建造者模式](https://www.runoob.com/wp-content/uploads/2014/08/builder_pattern_uml_diagram.jpg)
###### 用mealbuilder创建含有不同meal的对象

##### 创建步骤：
1. ##### 步骤 1
    创建一个表示食物条目和食物包装的接口。
2. ##### 步骤 2
    创建实现 Packing 接口的实体类。
3. ##### 步骤 3
    创建实现 Item 接口的抽象类，该类提供了默认的功能。
4. ##### 步骤 4
    创建扩展了 Burger 和 ColdDrink 的实体类。
5. ##### 步骤 5
    创建一个 Meal 类，带有上面定义的 Item 对象。
6. ##### 步骤 6
    创建一个 MealBuilder 类，实际的 builder 类负责创建 Meal 对象。
7. ##### 步骤 7
    BuiderPatternDemo 使用 MealBuider 来演示建造者模式（Builder Pattern）。
8. ##### 步骤 8
    执行程序，输出结果：


----


----
### 适配器模式
适配器模式（Adapter Pattern）是作为两个不兼容的接口之间的桥梁。这种类型的设计模式属于结构型模式，它结合了两个独立接口的功能。

##### 意图：将一个类的接口转换成客户希望的另外一个接口。适配器模式使得原本由于接口不兼容而不能一起工作的那些类可以一起工作。

##### 主要解决：主要解决在软件系统中，常常要将一些"现存的对象"放到新的环境中，而新环境要求的接口是现对象不能满足的。
##### 何时使用：一些基本部件不会变，而其组合经常变化的时候。
##### 优点： 1、可以让任何两个没有关联的类一起运行。 2、提高了类的复用。 3、增加了类的透明度。 4、灵活性好。

##### 缺点： 1、过多地使用适配器，会让系统非常零乱，不易整体进行把握。2.由于JAVA至多继承一个类，所以至多只能适配一个适配者类，而且目标类必须是抽象类。
##### 注意事项：适配器不是在详细设计时添加的，而是解决正在服役的项目的问题。

![适配器模式](https://www.runoob.com/wp-content/uploads/2014/08/adapter_pattern_uml_diagram.jpg)
###### 用audioPlayer调用，实现播放不同格式的文件

##### 创建步骤：
1. ##### 步骤 1
    为媒体播放器和更高级的媒体播放器创建接口：MediaPlayer.java，AdvancedMediaPlayer.java
2. ##### 步骤 2
    创建实现了 AdvancedMediaPlayer 接口的实体类：VlcPlayer.java，Mp4Player.java
3. ##### 步骤 3
    创建实现了 MediaPlayer 接口的适配器类：MediaAdapter.java
4. ##### 步骤 4
    创建实现了 MediaPlayer 接口的实体类：AudioPlayer.java
5. ##### 步骤 5
    使用 AudioPlayer 来播放不同类型的音频格式：AdapterPatternDemo.java
6. ##### 步骤 6
    执行程序，输出结果：

----

----    
### 单例模式
    这种类型的设计模式属于创建型模式，它提供了一种创建对象的最佳方式。
    这种模式涉及到一个单一的类，该类负责创建自己的对象，同时确保只有单个对象被创建。这个类提供了一种访问其唯一的对象的方式，可以直接访问，不需要实例化该类的对象。
##### 注意：
* 1、单例类只能有一个实例。
* 2、单例类必须自己创建自己的唯一实例。
* 3、单例类必须给所有其他对象提供这一实例。

##### 意图：保证一个类仅有一个实例，并提供一个访问它的全局访问点。
##### 主要解决：一个全局使用的类频繁地创建与销毁。
##### 何时使用：当您想控制实例数目，节省系统资源的时候。判断系统是否已经有这个单例，如果有则返回，如果没有则创建。
##### 优点： 1、在内存里只有一个实例，减少了内存的开销，尤其是频繁的创建和销毁实例（比如管理学院首页页面缓存）2、避免对资源的多重占用（比如写文件操作）。

##### 缺点： 没有接口，不能继承，与单一职责原则冲突，一个类应该只关心内部逻辑，而不关心外面怎么样来实例化。
##### 注意事项：getInstance() 方法中需要使用同步锁 synchronized (Singleton.class) 防止多线程同时进入造成 instance 被多次实例化。

![单例模式](https://www.runoob.com/wp-content/uploads/2014/08/62576915-36E0-4B67-B078-704699CA980A.jpg)
###### 构造函数为 private，这样该类就不会被实例化


#### 单例模式的实现
#### 懒汉模式是需要的时候创建对象，后面用到就判断有无现成的对象用，如果有则使用，没有则创建
#### 饿汉模式是在编译时就创建对象，不管用没用到。有可能产生浪费资源的情况。

![单例模式](https://www.runoob.com/wp-content/uploads/2014/08/62576915-36E0-4B67-B078-704699CA980A.jpg)

1. **懒汉式，线程不安全**
    **是否 Lazy 初始化**：是
    **是否多线程安全**：否
    **实现难度**：易
    **描述**：这种方式是最基本的实现方式，这种实现最大的问题就是不支持多线程。因为没有加锁synchronized，所以严格意义上它并不算单例模式。这种方式 lazy loading 很明显，不要求线程安全，在多线程不能正常工作。

        public class Singleton {  
            private static Singleton instance;  
            private Singleton (){}  
          
            public static Singleton getInstance() {  
            if (instance == null) {  
                instance = new Singleton();  
                }  
            return instance;  
            }  
        }



2. **懒汉式，线程安全**
**是否 Lazy 初始化**：是
**是否多线程全**：是
**实现难度**：易
**描述**：这种方式具备很好的 lazy loading，能够在多线程中很好的工作，但是，效率很低，99% 情况下不需要同步。
**优点**：第一次调用才初始化，避免内存浪费。
**缺点**：必须加锁 synchronized 才能保证单例，但加锁会影响效率。getInstance()的性能对应用程序不是很关键（该方法使用不太频繁）

        public class Singleton {  
            private static Singleton instance;  
            private Singleton (){}  
            public static synchronized Singleton getInstance() {  
            if (instance == null) {  
                instance = new Singleton();  
            }  
            return instance;  
            }  
        }

3. **饿汉式**
**是否 Lazy 初始化**：否
**是否多线程安全**：是
**实现难度**：易
**描述**：这种方式比较常用，但容易产生垃圾对象。
**优点**：没有加锁，执行效率会提高。
**缺点**：类加载时就初始化，浪费内存。
它基于 classloader 机制避免了多线程的同步问题，不过，instance在类装载时就实例化，虽然导致类装载的原因有很多种，在单例模式中大多数都是调用 getInstance 方法， 但是也不能确定有其他的方式（或者其他的静态方法）导致类装载，这时候初始化 instance 显然没有达到 lazy loading 的效果。

            public class Singleton {  
                private static Singleton instance = new Singleton();  
                private Singleton (){}  
                
                public static Singleton getInstance() {  
                return instance;  
                }  
            }

4. ##### 双检锁/双重校验锁（DCL，即 double-checked locking）
    **JDK 版本**：JDK1.5 起
    **是否 Lazy 初始化**：是
    **是否多线程安全**：是
    **实现难度**：较复杂
    **描述**：这种方式采用双锁机制，安全且在多线程情况下能保持高性能。getInstance() 的性能对应用程序很关键。

        public class Singleton {  
            private volatile static Singleton singleton;  
            private Singleton (){}  
            public static Singleton getSingleton() {  
            if (singleton == null) {  
                synchronized (Singleton.class) {  
                if (singleton == null) {  
                    singleton = new Singleton();  
                }  
                }  
            }  
            return singleton;  
            }  
        }

----

---
### 装饰器模式
装饰器模式（Decorator Pattern）允许向一个现有的对象添加新的功能，同时又不改变其结构。这种类型的设计模式属于结构型模式，它是作为现有的类的一个包装。
这种模式创建了一个装饰类，用来包装原有的类，并在保持类方法签名完整性的前提下，提供了额外的功能。
我们通过下面的实例来演示装饰器模式的用法。其中，我们将把一个形状装饰上不同的颜色，同时又不改变形状类。

##### 意图：动态地给一个对象添加一些额外的职责。就增加功能来说，装饰器模式相比生成子类更为灵活。
##### 主要解决：一般的，我们为了扩展一个类经常使用继承方式实现，由于继承为类引入静态特征，并且随着扩展功能的增多，子类会很膨胀
##### 何时使用：在不想增加很多子类的情况下扩展类。
##### 优点： 装饰类和被装饰类可以独立发展，不会相互耦合，装饰模式是继承的一个替代模式，装饰模式可以动态扩展一个实现类的功能。
##### 缺点：多层装饰比较复杂。
##### 注意事项：可代替继承。

![装饰器模式](https://www.runoob.com/wp-content/uploads/2014/08/decorator_pattern_uml_diagram.jpg)

##### 创建步骤：
1. ##### 步骤 1
    创建一个接口：Shape.java
2. ##### 步骤 2
    创建实现接口的实体类：Rectangle.java，Circle.java
3. ##### 步骤 3
    创建实现了 Shape 接口的抽象装饰类：ShapeDecorator.java
4. ##### 步骤 4
    创建扩展了 ShapeDecorator 类的实体装饰类：RedShapeDecorator.java
5. ##### 步骤 5
    使用 RedShapeDecorator 来装饰 Shape 对象：DecoratorPatternDemo.java
6. ##### 步骤 6
    执行程序，输出结果

---

---
* [设计模式参考] (https://www.runoob.com/design-pattern/design-pattern-tutorial.html)
---

