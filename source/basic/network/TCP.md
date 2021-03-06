## TCP 的特性

* TCP 提供一种**面向连接的、可靠的**字节流服务
* 在一个 TCP 连接中，仅有两方进行彼此通信。广播和多播不能用于 TCP
* TCP 使用校验和，确认和重传机制来保证可靠传输
* TCP 给数据分节进行排序，并使用累积确认保证数据的顺序不变和非重复
* TCP 使用滑动窗口机制来实现流量控制，通过动态改变窗口的大小进行拥塞控制

**注意**：TCP 并不能保证数据一定会被对方接收到，因为这是不可能的。TCP 能够做到的是，如果有可能，就把数据递送到接收方，否则就（通过放弃重传并且中断连接这一手段）通知用户。因此准确说 TCP 也不是 100% 可靠的协议，它所能提供的是数据的可靠递送或故障的可靠通知。

## 三次握手与四次挥手

所谓三次握手(Three-way Handshake)，是指建立一个 TCP 连接时，需要客户端和服务器总共发送3个包。

三次握手的目的是连接服务器指定端口，建立 TCP 连接，并同步连接双方的序列号和确认号，交换 TCP 窗口大小信息。在 socket 编程中，客户端执行 `connect()` 时。将触发三次握手。
    
    
* 第一次握手(SYN=1, seq=x):
    
   客户端发送一个 TCP 的 SYN 标志位置1的包，指明客户端打算连接的服务器的端口，以及初始序号 X,保存在包头的序列号(Sequence Number)字段里。

   发送完毕后，客户端进入 `SYN_SEND` 状态。
   
* 第二次握手(SYN=1, ACK=1, seq=y, ACKnum=x+1):
    
   服务器发回确认包(ACK)应答。即 SYN 标志位和 ACK 标志位均为1。服务器端选择自己 ISN 序列号，放到 Seq 域里，同时将确认序号(Acknowledgement Number)设置为客户的 ISN 加1，即X+1。 
   发送完毕后，服务器端进入 `SYN_RCVD` 状态。

* 第三次握手(ACK=1，ACKnum=y+1)
   
   客户端再次发送确认包(ACK)，SYN 标志位为0，ACK 标志位为1，并且把服务器发来 ACK 的序号字段+1，放在确定字段中发送给对方，并且在数据段放写ISN的+1
   
   发送完毕后，客户端进入 `ESTABLISHED` 状态，当服务器端接收到这个包时，也进入 `ESTABLISHED` 状态，TCP 握手结束。

三次握手的过程的示意图如下：

![three-way-handshake](https://raw.githubusercontent.com/HIT-Alibaba/interview/master/img/tcp-connection-made-three-way-handshake.png)

TCP 的连接的拆除需要发送四个包，因此称为四次挥手(Four-way handshake)，也叫做改进的三次握手。客户端或服务器均可主动发起挥手动作，在 socket 编程中，任何一方执行 `close()` 操作即可产生挥手操作。

* 第一次挥手(FIN=1，seq=x)
   
   假设客户端想要关闭连接，客户端发送一个 FIN 标志位置为1的包，表示自己已经没有数据可以发送了，但是仍然可以接受数据。
   
   发送完毕后，客户端进入 `FIN_WAIT_1` 状态。
   
* 第二次挥手(ACK=1，ACKnum=x+1)
    
   服务器端确认客户端的 FIN 包，发送一个确认包，表明自己接受到了客户端关闭连接的请求，但还没有准备好关闭连接。
   
   发送完毕后，服务器端进入 `CLOSE_WAIT` 状态，客户端接收到这个确认包之后，进入 `FIN_WAIT_2` 状态，等待服务器端关闭连接。
   
* 第三次挥手(FIN=1，seq=y)

   服务器端准备好关闭连接时，向客户端发送结束连接请求，FIN 置为1。
   
   发送完毕后，服务器端进入 `LAST_ACK` 状态，等待来自客户端的最后一个ACK。
   
* 第四次挥手(ACK=1，ACKnum=y+1)
    
   客户端接收到来自服务器端的关闭请求，发送一个确认包，并进入 `TIME_WAIT `状态，等待可能出现的要求重传的 ACK 包。
   
   服务器端接收到这个确认包之后，关闭连接，进入 `CLOSED` 状态。
   
   客户端等待了某个固定时间（两个最大段生命周期，2MSL，2 Maximum Segment Lifetime）之后，没有收到服务器端的 ACK ，认为服务器端已经正常关闭连接，于是自己也关闭连接，进入 `CLOSED` 状态。
        
四次挥手的示意图如下：

![four-way-handshake](https://raw.githubusercontent.com/HIT-Alibaba/interview/master/img/tcp-connection-closed-four-way-handshake.png)

## TCP 协议如何保证可靠传输
1. 应用数据被分割成 TCP 认为最适合发送的数据块。
2. TCP 给发送的每一个包进行编号，接收方对数据包进行排序，把有序数据传送给应用层。
3. 校验和： TCP 将保持它首部和数据的检验和。这是一个端到端的检验和，目的是检测数据在传输过程中的任何变化。如果收到段的检验和有差错，TCP 将丢弃这个报文段和不确认收到此报文段。
4. TCP 的接收端会丢弃重复的数据。
5. **流量控制**： TCP 连接的每一方都有固定大小的缓冲空间，TCP的接收端只允许发送端发送接收端缓冲区能接纳的数据。当接收方来不及处理发送方的数据，能提示发送方降低发送的速率，防止包丢失。TCP使用的流量控制协议是可变大小的滑动窗口协议。（TCP利用滑动窗口实现流量控制）
6. **拥塞控制**： 当网络拥塞时，减少数据的发送。
7. **停止等待协议** 也是为了实现可靠传输的，它的基本原理就是每发完一个分组就停止发送，等待对方确认。在收到确认后再发下一个分组。 
8. **超时重传**： 当 TCP 发出一个段后，它启动一个定时器，等待目的端确认收到这个报文段。如果不能及时收到一个确认，将重发这个报文段。


## 拥塞控制和流量控制不同，拥塞控制是一个全局性的过程，流量控制指点对点通信量的控制。

## TCP的流量控制
(滑动窗口)  TCP 采用大小可变的滑动窗口进行流量控制(窗口大小的单位是字节),在TCP报文段首部的窗口字段写入的数值就是当前给对方设置的发送窗口数值的上限。发送窗口在连接建立时由双方商定。但在通信的过程中,接收端可根据自己的资源情况,随时动态地调整对方的发送窗口上限值(可增大或减小)。
* 控制过程:
    在通信过程中,接收方根据自己接收缓存的大小,动态地调整发送方的发送窗口大小,这就是接收窗口rwnd,即调整TCP报文段首部中的“窗口”字段值,来限制发送方向网络注入报文的速率。同时,发送方根据其对当前网络拥塞程序的估计而确定的窗口值,称为拥塞窗口cwnd,其大小与网络的带宽和时延密切相关。发送窗口的实际大小是取rwnd和cwnd中的最小值。


## TCP阻塞控制
在某段时间，若对网络中某一资源的需求超过了该资源所能提供的可用部分，网络的性能就会变坏，这种情况就叫做拥塞。
拥塞控制就是 防止过多的数据注入网络中，这样可以使网络中的路由器或链路不致过载。
拥塞控制的方法主要有以下四种：
*   1.慢启动：不要一开始就发送大量的数据，先探测一下网络的拥塞程度，也就是说由小到大逐渐增加拥塞窗口的大小
*   2.拥塞避免：
*   3.快重传：网络拥塞时的处理
*   4.快恢复：网络拥塞时的处理，快重传配合使用的还有快恢复算法


## 正向代理和反向代理

#### 正向代理就是我们日常用的代理，通过代理我们本机去访问服务器。
#### 反向代理刚好相反，客户端是感觉不到反向代理的存在的。反向代理代理服务器，代替服务器接收访问请求，然后从服务器取到内容，再返还回给客户端。反向代理可以用于实现负载均衡，保证内网安全等。


-------


## SYN攻击

* 什么是 SYN 攻击（SYN Flood）？

    在三次握手过程中，服务器发送 SYN-ACK 之后，收到客户端的 ACK 之前的 TCP 连接称为半连接(half-open connect)。此时服务器处于 SYN_RCVD 状态。当收到 ACK 后，服务器才能转入 ESTABLISHED 状态.

    SYN 攻击指的是，攻击客户端在短时间内伪造大量不存在的IP地址，向服务器不断地发送SYN包，服务器回复确认包，并等待客户的确认。由于源地址是不存在的，服务器需要不断的重发直至超时，这些伪造的SYN包将长时间占用未连接队列，正常的SYN请求被丢弃，导致目标系统运行缓慢，严重者会引起网络堵塞甚至系统瘫痪。
    
    SYN 攻击是一种典型的 DoS/DDoS 攻击。
    
* 如何检测 SYN 攻击？

     检测 SYN 攻击非常的方便，当你在服务器上看到大量的半连接状态时，特别是源IP地址是随机的，基本上可以断定这是一次SYN攻击。在 Linux/Unix 上可以使用系统自带的 netstats 命令来检测 SYN 攻击。
     
* 如何防御 SYN 攻击？

    SYN攻击不能完全被阻止，除非将TCP协议重新设计。我们所做的是尽可能的减轻SYN攻击的危害，常见的防御 SYN 攻击的方法有如下几种：
    
    * 缩短超时（SYN Timeout）时间
    * 增加最大半连接数
    * 过滤网关防护
    * SYN cookies技术
   
## TCP KeepAlive

TCP 的连接，实际上是一种纯软件层面的概念，在物理层面并没有“连接”这种概念。TCP 通信双方建立交互的连接，但是并不是一直存在数据交互，有些连接会在数据交互完毕后，主动释放连接，而有些不会。在长时间无数据交互的时间段内，交互双方都有可能出现掉电、死机、异常重启等各种意外，当这些意外发生之后，这些 TCP 连接并未来得及正常释放，在软件层面上，连接的另一方并不知道对端的情况，它会一直维护这个连接，长时间的积累会导致非常多的半打开连接，造成端系统资源的消耗和浪费，为了解决这个问题，在传输层可以利用 TCP 的 KeepAlive 机制实现来实现。主流的操作系统基本都在内核里支持了这个特性。

TCP KeepAlive 的基本原理是，隔一段时间给连接对端发送一个探测包，如果收到对方回应的 ACK，则认为连接还是存活的，在超过一定重试次数之后还是没有收到对方的回应，则丢弃该 TCP 连接。

[TCP-Keepalive-HOWTO](http://www.tldp.org/HOWTO/html_single/TCP-Keepalive-HOWTO/) 有对 TCP KeepAlive 特性的详细介绍，有兴趣的同学可以参考。这里主要说一下，TCP KeepAlive 的局限。首先 TCP KeepAlive 监测的方式是发送一个 probe 包，会给网络带来额外的流量，另外 TCP KeepAlive 只能在内核层级监测连接的存活与否，而连接的存活不一定代表服务的可用。例如当一个服务器 CPU 进程服务器占用达到 100%，已经卡死不能响应请求了，此时 TCP KeepAlive 依然会认为连接是存活的。因此 TCP KeepAlive 对于应用层程序的价值是相对较小的。需要做连接保活的应用层程序，例如 QQ，往往会在应用层实现自己的心跳功能。

### 参考资料

* 计算机网络：自顶向下方法
* [TCP三次握手及四次挥手详细图解](http://www.cnblogs.com/hnrainll/archive/2011/10/14/2212415.html)
* [TCP协议三次握手过程分析](http://www.cnblogs.com/rootq/articles/1377355.html)
* [TCP协议中的三次握手和四次挥手(图解)](http://blog.csdn.net/whuslei/article/details/6667471)
* [百度百科：SYN攻击](http://baike.baidu.com/subview/32754/8048820.htm)
* [TCP-Keepalive-HOWTO](http://www.tldp.org/HOWTO/html_single/TCP-Keepalive-HOWTO/)


