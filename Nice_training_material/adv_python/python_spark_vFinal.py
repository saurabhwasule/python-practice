"""
Installation 11/4
    hadoop and spark only (no hive, use spark local hive at first)
    standalone app
        WordCount
    Scaling-up
        spark-submit
        yarn
        master/worker
    spark-shell/pyspark
DenseVector and SparseVector (mlib data types) 12/4
RDD Basics  13/4
DataFrames and operations  14/4
DataSource - csv, json, paraquet and Hive 17/4
Stats 18/4
    MLIB - basic, chisquare test 
ML(not MLIB) 
    General Flow 19/4
    Pipeline     19/4
    Feature extraction /seletion - Tokenizer, StringIndex,.., PCA  19/4
    Evaluator    20/4
    Classifications - Random Forest and Gradient Boost 21/4
    Regression - Random Forest and Gradient Boost    21/4
    Regression - LM and GLM  21/4    
    Clustering - KMeans  24/4
    Streaming - Kmeans   24/4
    Streaming - Regression 
"""

###*** Note 
#To use python2.7, use below at first 
set PATH=C:\Anaconda2;C:\Anaconda2\Scripts;%PATH%
#Note numpy, pandas are preferred to be installed 

#Start pyspark or submit any file 
$ set SPARK_HOME=<path to Spark home>
$ set PYTHONPATH=%SPARK_HOME%/python/lib/py4j-0.10.4-src.zip;%SPARK_HOME%/python/;%PYTHONPATH%

$ pyspark --master local[4] --driver-memory 2G
#sc(spark context) and spark(spark session) are autocreated 

#or submit 
$ spark-submit --master local[4] --driver-memory 2G script.py args 
#with script starting as 
conf = SparkConf().setAppName("NAME") 
sc = SparkContext(conf=conf)

#or run directly 
$ python script.py args 
#with script starting as 
conf = SparkConf().setAppName("wordCount").setMaster(master_url) 
sc = SparkContext(conf=conf)

#Check application web UI at http://192.168.1.4:4040(http://<driver>:4040 )
#check Spark properties in the 'Environment' tab
#spark properties are at conf/spark-defaults.conf, SparkConf(), or the command line will appear.
#Help of properties: https://spark.apache.org/docs/latest/configuration.html#available-properties




###Spark - Introduction 

#RDD consists of many partitions, 
#with each partition on a core of a node in cluster 

#One Partition executes one Task 

#One program consists of many stages 

#A stage is consisting of many parallel tasks 
#where one task is on one  partition 

#stage is uniquely identified by  id . 
#When a stage is created, 
#DAGScheduler increments internal counter  nextStageId  to track the number of stage submissions.

#stage can only work on the partitions of a single RDD (identified by  rdd ), 
#but can be associated with many other dependent parent stages (via internal field  parents ), 
#with the boundary of a stage marked by shuffle dependencies.       
        
        
#There are two types of stages:
•ShuffleMapStage is an intermediate stage (in the execution DAG) that produces data for other stage(s). 
 It writes map output files for a shuffle. 
•ResultStage is the final stage that executes a Spark action in a user program by running a function on an RDD.

#Submitting a stage can therefore trigger execution of a series of dependent parent stages   

#Display means 
[Stage7:===========>                              (14174 + 5) / 62500]        
        
#Stage 7: shows the stage you are in now, 
#(14174 + 5) / 62500] is (numCompletedTasks + numActiveTasks) / totalNumOfTasksInThisStage 

#The progress bar shows numCompletedTasks / totalNumOfTasksInThisStage.

#shown when both spark.ui.showConsoleProgress is true (by default) 
#and log level in conf/log4j.properties is ERROR or WARN (!log.isInfoEnabled is true).
       
        
###Spark - Introduction - partition (aka split) 
#a logical chunk of a large distributed data set, called RDD (can be imagined as rows of elements, ie 1D, 2D etc )  
#Spark manages data using partitions that helps parallelize distributed data processing 
#with minimal network traffic for sending data between executors.


#By default, Spark tries to read data into an RDD from the nodes that are close to it.
#By default, a partition is created for each HDFS partition, which by default is 64MB 

     
#For example 
#Note all Higher order functions are lazy, to force, use some Action at the end 
>>> rdd = sc.parallelize(range(100)) #default no of partitions = no of cores 
>>> rdd.count()
100 
>>> rdd.getNumPartitions()
4
        
#Manually modifying 
>>> ints = sc.parallelize(range(100), 2) #two partitions 
>>> ints.getNumPartitions()
2
#check patition with values 
>>> ints.mapPartitionsWithIndex(lambda index,values : (index, list(values))).col
[0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 2
1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 4
1, 42, 43, 44, 45, 46, 47, 48, 49], 1, [50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]


#smaller/more numerous partitions allow work to be distributed among more workers, 
#but larger/fewer partitions allow work to be done in larger chunks,
#which may result in the work getting done more quickly as long 
#as all workers are kept busy, due to reduced overhead.
        

#Spark can only run 1 task for every partition(because one partition=onecore) of an RDD, 
#up to the number of cores in your cluster concurrently 

#you generally want at least as many as the number of executors for parallelism. 
#You can get this computed value by calling  
>>> sc.defaultParallelism  #no of cores 
4
 


#For compressed file, number of partion is always 1 as .gz etc can not work in parallel
#After reading change the partition via repartition or coalesce

def repartition(numPartitions: Int): RDD[T]
#repartition  is coalesce with  numPartitions  and  shuffle  enabled.      
def coalesce(numPartitions: Int, shuffle: Boolean = false): RDD[T]

rdd = sc.parallelize(range(10), 8)
>>> rdd.getNumPartitions()
8

res1 = rdd.coalesce(numPartitions=8, shuffle=False)  
>>> res1.toDebugString()
(8) CoalescedRDD[1] at coalesce at <console>:27 []
 |  ParallelCollectionRDD[0] at parallelize at <console>:24 []

res3 = rdd.coalesce(numPartitions=8, shuffle=True)
>>> res3.toDebugString()
res4: String =
(8) MapPartitionsRDD[5] at coalesce at <console>:27 []
 |  CoalescedRDD[4] at coalesce at <console>:27 []
 |  ShuffledRDD[3] at coalesce at <console>:27 []
 +-(8) MapPartitionsRDD[2] at coalesce at <console>:27 []
    |  ParallelCollectionRDD[0] at parallelize at <console>:24 []
      
      
      

      
/***** Installation and Standalone applications ****/

###Hadoop and spark installation 
#Linux: https:#hadoop.apache.org/docs/r2.7.3/hadoop-project-dist/hadoop-common/SingleCluster.html
#Windows 
#REF: https://mariuszprzydatek.com/2015/05/10/installing_hadoop_on_windows_8_or_8_1/

1. download binary from below 
#Windows Binary: https://github.com/karthikj1/Hadoop-2.7.1-Windows-64-binaries/releases/download/v2.7.1/hadoop-2.7.1.tar.gz
#spark : https://archive.apache.org/dist/spark/spark-2.1.0/spark-2.1.0-bin-hadoop2.7.tgz

2. unzip hadoop to c:\hadoop and spark to c:\spark 

3. ENV var , HADOOP_HOME=c:\hadoop and SPARK_HOME=c:\spark
And add to PATH , c:\hadoop\bin;c:\hadoop\sbin;c:\spark\bin;c:\spark\sbin

4. Copy JDK to c:\hadoop\java eg to c:\hadoop\java\jdk1.8.0_65

5. Update below files in c:\hadoop\etc\hadoop - for pseudo cluster mode 
#hadoop-env.cmd: begining add and replace if existing 
set HADOOP_PREFIX=%HADOOP_HOME%
set JAVA_HOME=%HADOOP_HOME%\java\jdk1.8.0_65
set HADOOP_CONF_DIR=%HADOOP_PREFIX%\etc\hadoop
set YARN_CONF_DIR=%HADOOP_CONF_DIR%
set PATH=%PATH%;%HADOOP_PREFIX%\bin

#https://hadoop.apache.org/docs/r2.7.1/hadoop-project-dist/hadoop-common/SingleCluster.html
##Standalone Operation
#By default, Hadoop is configured to run in a non-distributed mode, 
#as a single Java process. This is useful for debugging.


##Pseudo-Distributed Operation 
#change core-site.xml , hdfs-site.xml etc(note below xml changes to default to yarn, hence start yarn)
#Hadoop can also be run on a single-node in a pseudo-distributed mode 
#where each Hadoop daemon runs in a separate Java process.

#all are in admin console 
#1.Format the filesystem:
  $ hdfs namenode -format

#2.Start NameNode daemon and DataNode daemon:
  $ start-dfs & start-yarn
  #start history server if required to check logs 
  $ mapred --config c:/hadoop/etc/hadoop historyserver
  #linux
  $ sbin/mr-jobhistory-daemon.sh --config $HADOOP_CONFIG_DIR start historyserver

#check all process started 
  $ jps 
  
#The hadoop daemon log output is written to the $HADOOP_LOG_DIR directory 
#(defaults to $HADOOP_HOME/logs).


#3.Browse the web interface for the NameNode; by default it is available at:
#◦NameNode - http://localhost:50070/

##Example of running mapReduce Job 
#4.Make the HDFS directories required to execute MapReduce jobs:
  $ hdfs dfs -mkdir /user
  $ hdfs dfs -mkdir /user/<username>

#5.Copy the input files into the distributed filesystem:
  $ cd  c:\hadoop
  $ hdfs dfs -put etc/hadoop input   #input means /user/das/input 
  $ hdfs dfs -ls input 
  $ hdfs dfs -ls hdfs://localhost:19000/user/das/input  #as per core-site.xml

#6.Run some of the examples provided:
  $ hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar grep input output "dfs[a-z.]+"



#7.Examine the output files: Copy the output files from the distributed filesystem to the local filesystem and examine them:
  $ hdfs dfs -get output output
  $ cat output/*
#or   
  $ hdfs dfs -cat output/*

#8.When you're done, stop the daemons with: 
  $ sop-yarn & stop-dfs

##All configuration files 
#hdfs-site.xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
</configuration>
#core-site.xml
<configuration>
  <property>
    <name>fs.default.name</name>
    <value>hdfs://0.0.0.0:19000</value>
  </property>
</configuration>
#slaves - list of slaves 
localhost
#mapred-site.xml
<configuration>
  <property>
    <name>mapreduce.job.user.name</name>
    <value>das</value>
  </property>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
  <property>
    <name>yarn.apps.stagingDir</name>
    <value>/user/das/staging</value>
  </property>
  <property>
    <name>mapreduce.jobtracker.address</name>
    <value>local</value>
  </property>
  <property>
      <name>mapreduce.jobhistory.address</name>
      <value>localhost:10020</value>
    </property>    
    <property>
        <name>mapreduce.jobhistory.webapp.address</name>
        <value>localhost:19888</value>
    </property>
</configuration>
#yarn-site.xml
<configuration>
  <property>
    <name>yarn.server.resourcemanager.address</name>
    <value>0.0.0.0:8020</value>
  </property>
  <property>
    <name>yarn.server.resourcemanager.application.expiry.interval</name>
    <value>60000</value>
  </property>
  <property>
    <name>yarn.server.nodemanager.address</name>
    <value>0.0.0.0:45454</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
    <value>org.apache.hadoop.mapred.ShuffleHandler</value>
  </property>
  <property>
    <name>yarn.server.nodemanager.remote-app-log-dir</name>
    <value>/app-logs</value>
  </property>
  <property>
    <name>yarn.nodemanager.log-dirs</name>
    <value>/dep/logs/userlogs</value>
  </property>
  <property>
         <name>yarn.log.server.url</name>
         <value>http://localhost:19888/jobhistory/logs</value>
  </property>
  <property>
    <name>yarn.server.mapreduce-appmanager.attempt-listener.bindAddress</name>
    <value>0.0.0.0</value>
  </property>
  <property>
    <name>yarn.server.mapreduce-appmanager.client-service.bindAddress</name>
    <value>0.0.0.0</value>
  </property>
  <property>
    <name>yarn.log-aggregation-enable</name>
    <value>true</value>
  </property>
  <property>
    <name>yarn.log-aggregation.retain-seconds</name>
    <value>-1</value>
  </property>
  <property>
    <name>yarn.application.classpath</name>
    <value>%HADOOP_CONF_DIR%,%HADOOP_HOME%/share/hadoop/common/*,%HADOOP_HOME%/share/hadoop/common/lib/*,%HADOOP_HOME%/share/hadoop/hdfs/*,%HADOOP_HOME%/share/hadoop/hdfs/lib/*,%HADOOP_HOME%/share/hadoop/mapreduce/*,%HADOOP_HOME%/share/hadoop/mapreduce/lib/*,%HADOOP_HOME%/share/hadoop/yarn/*,%HADOOP_HOME%/share/hadoop/yarn/lib/*,%SPARK_HOME%/lib/*</value>   <!-- */ -->
  </property>
  <property> 
      <name>yarn.nodemanager.delete.debug-delay-sec</name> 
      <value>600</value> 
  </property>

#Run an example YARN job 
cd  c:\hadoop
hadoop fs -mkdir  example
hadoop fs -put license.txt  example/
hadoop fs -ls example/
 
#Run an example YARN job 
yarn jar share\hadoop\mapreduce\hadoop-mapreduce-examples-*.jar wordcount example/license.txt out
 
#9. Check delete 
hadoop fs -ls out
hadoop fs -rm -r -f  out


###*Checking: Check the following pages in your browser: 
Resource Manager:  http://localhost:8088
Web UI of the NameNode daemon:  http://localhost:50070
HDFS NameNode web interface:  http://localhost:8042

###* Debugging
#To review per-container launch environment, 
#increase yarn.nodemanager.delete.debug-delay-sec to a large value (e.g. 36000), 
#yarn-site.xml
<property> 
      <name>yarn.nodemanager.delete.debug-delay-sec</name> 
      <value>3600</value> 
</property>
#then access the application cache through 
#yarn.nodemanager.local-dirs on the nodes on which containers are launched

#By default - yarn.nodemanager.local-dirs is ${hadoop.tmp.dir}/nm-local-dir ie c:/tmp/nm-local-dir

#An applications localized file directory will be found in: 
#${yarn.nodemanager.local-dirs}/usercache/${user}/appcache/application_${appid}. 
#Individual containers work directories, called container_${contid}, will be subdirectories of this. 

###* To leave safe mode (maintenance mode)
hdfs dfsadmin -safemode get
hdfs dfsadmin -safemode leave

###* check yarn logs 
#find application id from yarn-resourcemanager cmd shell windows 
$ yarn logs -applicationId application_1487592307594_0001 > log.txt

#check logs from yarn.nodemanager.log-dirs of yarn-site.xml
#as per above it is c:/deps/logs/userlogs/...

##Linux: Setup passphraseless ssh

###* for linux to start all nodes
#must ssh to the localhost without a passphrase:
  $ ssh localhost
#If you cannot ssh to localhost without a passphrase, execute the following commands:
  $ ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa
  $ cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
  $ export HADOOP\_PREFIX=/usr/local/hadoop


  
  
###Spark:  Configure Spark 
#Starting version 2.0, Spark is built with Scala 2.11 by default. 

#do below (after creating new tmp\hive), Note spak-shell requires /tmp/hive at root 
#hence if you are starting at c:, then do below for c:\tmp\hive 
hadoop\bin\winutils.exe chmod 777 c:\tmp\hive 
#check at c:(where /tmp/hive is created)
$ pyspark 

##Spark - conf 
#To use HDFS : set HADOOP_CONF_DIR in $SPARK_HOME/spark-env.cmd or .sh 

#Check application web UI at http://192.168.1.4:4040(http://<driver>:4040 )
#check Spark properties in the 'Environment' tab
#spark properties are at conf/spark-defaults.conf, SparkConf(), or the command line will appear.
#Help of properties: https://spark.apache.org/docs/latest/configuration.html#available-properties

#Properties set directly on the SparkConf take highest precedence
sparkConf.set(key: String, value: String): SparkConf 
#then flags passed to spark-submit or pyspark(via --conf )
#then options in the spark-defaults.conf file

#Various configurations - check conf/*.template file for templates 
spark-defaults.conf     Spark properties default values 
spark-env.sh/cmd        Various environment variables
log4j.properties        Logging properties 


##Spark - Logging 
#Edit  conf/log4j.properties file and change the following line:
log4j.rootCategory=INFO, console
#to
log4j.rootCategory=ERROR, console
#programatically , use sparkContext 's method 
sc.setLogLevel(logLevel: String): Unit 
#logLevel  The desired log level as a string. 
#Valid log levels include: ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN


##Spark -  standalone cluster  setup 
#Master and worker start in admin console 
#start master (not same as Driver - Driver contains 'main' program of user)
#master assigns job to Worker/Executor when deploy-mode=client 
#make %SPARK_HOME/conf/log4j.properties:log4j.rootCategory=INFO to check console outputs 
spark-class2 org.apache.spark.deploy.master.Master 

#logs 
Master: Starting Spark master at spark://192.168.1.2:7077
Master: Running Spark version x.x.x
Utils: Successfully started service 'MasterUI' on port 8080.
Started MasterWebUI at http://192.168.1.2:8080


##Spark - worker/executor start , in all machines - check ip, 192.158.1.2 
spark-class2 org.apache.spark.deploy.worker.Worker spark://192.168.1.2:7077 
#Repeat above for all the slaves nodes 




###Spark -  Self-Contained Applications

#--py-files argument of spark-submit to add .py, .zip or .egg files (eg external library)
#to be distributed with your application

$ spark-submit \  
  --master <master-url> \
  --deploy-mode <deploy-mode> \
  --conf <key>=<value> \
  --py-files path/to/my/egg.egg \
  app.py arg1 arg2

  
  
#Spark - Master URL
local     Run Spark locally with one worker thread (i.e. no parallelism at all).  
local[K]  Run Spark locally with K worker threads (ideally, set this to the number of cores on your machine).  
local[*]  Run Spark locally with as many worker threads as logical cores on your machine. 
spark://HOST:PORT  Connect to the given Spark standalone cluster master. The port must be whichever one your master is configured to use, which is 7077 by default.  
mesos://HOST:PORT  Connect to the given Mesos cluster. The port must be whichever one your is configured to use, which is 5050 by default. Or, for a Mesos cluster using ZooKeeper, use mesos:#zk:#.... To submit with --deploy-mode cluster, the HOST:PORT should be configured to connect to the MesosClusterDispatcher.  
yarn      Connect to a  YARN  cluster in client or cluster mode depending on the value of --deploy-mode. 
          The cluster location will be found based on the HADOOP_CONF_DIR or YARN_CONF_DIR variable 

#when deploy-mode is client, 
#You could run spark-submit on laptop, and the Driver Program would run on that laptop. 

#when deploy-mode is cluster, 
#then cluster manager (YARN, MESOS) is used to find a slave of yarn/mesos-cluster
#having enough available resources to execute the Driver Program. 

#As a result, the Driver Program would run on one of the slave nodes of that yarn/mesos-cluster
#As its execution is delegated, you can not get the result from Driver Program, 
#it must store its results in a file, database, etc



#Example code:wordcount.py
from __future__ import print_function

from pyspark import SparkConf, SparkContext

if __name__ == "__main__":
    inputFile = "hdfs://localhost:19000/user/das/README"
      
    conf = SparkConf().setAppName("wordCount") #.setMaster(master_url) #when running without spark-submit
    #Create a Scala Spark Context.
    sc = SparkContext(conf=conf)
    #Load our input data.
    input =  sc.textFile(inputFile)
    #Split up into words.
    words = input.flatMap(lambda line : line.split(" "))
    #Transform into word and count.
    counts = words.map(lambda word : (word, 1)).reduceByKey(lambda v1, v2 : v1 + v2) #same key, func for values 
      
    #print, returns list
    output = counts.collect()
    for word, count in output:
        print("%s, %i" % (word, count)) 






#usage 
$ hdfs dfs -put README  /user/das/README
#set python2 as we have numpy installed there 
$ set PATH=c:\anaconda2;%PATH%

##Spark - Usage - local mode 
# add Python .zip, .egg or .py files to the search path with '--py-files'
#these files would be distributed to all 
$ spark-submit --master local[4] wordcount.py

##Spark -  In Master/workers mode (admin console)
$ spark-class2 org.apache.spark.deploy.master.Master 
$ spark-class2 org.apache.spark.deploy.worker.Worker spark://192.168.1.2:7077 

#usage - master/workr mode (localhost/127.0.0.1 etc does not work)
$ spark-submit  --master spark://192.168.1.2:7077 wordcount.py

##Spark -  Yarn mode (yarn should be running)
$ set HADOOP_CONF_DIR=c:\hadoop\etc\hadoop\conf
$ spark-submit --master yarn --deploy-mode client wordcount.py

#check logs 
http://home-e402:8088/cluster/app/application_1491884936563_0005/ 
#get Application_id from spark-submit log


##Spark -   cluster mode, no stdout
$ spark-submit --master yarn --deploy-mode cluster wordcount.py

#so check output from 
http://home-e402:8088/cluster/app/application_1491884936563_0007/ 
#get Application_id from spark-submit log






##Spark -   How to add jars 
#Check spark cluster concepts 
http://spark.apache.org/docs/latest/cluster-overview.html

#Spark applications run as independent sets of processes on a cluster, 
#coordinated by the SparkContext object in main program (called the driver program).

#The system currently supports three cluster managers:
•Standalone – a simple cluster manager included with Spark that makes it easy to set up a cluster.
•Apache Mesos – a general cluster manager that can also run Hadoop MapReduce and service applications.
•Hadoop YARN – the resource manager in Hadoop 2.


#When using spark-submit, 
#the application jar along with any jars included with the --jars option will be automatically transferred to the cluster. 
#URLs supplied after --jars must be separated by commas. 
#this list is included on the driver and executor classpaths(new in 2.1.0, earlier - NO)
#Directory expansion does not work with --jars.

#For Python, the equivalent --py-files option can be used 
#to distribute .egg, .zip and .py libraries to executors

#Spark uses the following URL scheme 
•file:          Absolute paths and file:/ URIs are served by the driver's HTTP file server, 
                and every executor pulls the file from the driver HTTP server.
•hdfs:, http:
https:, ftp:    These pull down files and JARs from the URI as expected
•local:         a URI starting with local:/ is expected to exist as a local file on each worker node. 
                This means that no network IO will be incurred, 
                and works well for large files/JARs that are pushed to each worker, 
                or shared via NFS, GlusterFS, etc.

#before spark 2.0 , add all these options to get jars to Driver(main program) and Executor(worker)
spark-submit --jars additional1.jar,additional2.jar \
  --driver-class-path additional1.jar:additional2.jar \
  --conf spark.executor.extraClassPath=additional1.jar:additional2.jar \
  --master <master-url> \
  --deploy-mode <deploy-mode> \
  --conf <key>=<value> \
  --py-files path/to/my/egg.egg \
  app.py arg1 arg2



##Saprk - SparkContext also has below methods 
#path supports all URL schems 
addPyFile(path)
    Add a .py or .zip dependency for all tasks to be executed 
    on this SparkContext in the future.
addFile(path, recursive=False)
    Add a file to be downloaded with this Spark job on every node
    To access the file, use SparkFiles.get(fileName)
    Currently directories(recursive=True) are only supported for Hadoop-supported filesystems
#Example 
from pyspark import SparkFiles
path = os.path.join(tempdir, "test.txt")
with open(path, "w") as testFile:
   _ = testFile.write("100")

sc.addFile(path)
def func(iterator):
   with open(SparkFiles.get("test.txt")) as testFile:
       fileVal = int(testFile.readline())
       return [x * fileVal for x in iterator]

>>> sc.parallelize([1, 2, 3, 4]).mapPartitions(func).collect()
[100, 200, 300, 400]
    
    
    

###Spark -    Few  ERRORs 
#*Error - Task is not serializable- 
#RDD[T] T must be Serializable
#And or any transformation function on RDD must use Serializable objects inside function
#Refer to RDD basics below 



#*Error - below for memory consumed, hence increase overhead 
17/02/20 17:02:52 INFO scheduler.DAGScheduler: ResultStage 71 (collect at BinaryClassificationMetrics.scala:19
2) failed in 15.100 s due to org.apache.spark.shuffle.FetchFailedException: Connection from /192.168.1.2:32790 closed
 
#solution
--conf spark.yarn.executor.memoryOverhead=1G
 
#*Error - Caused by multiple netty versions on classpath 
Caused by: java.lang.NoSuchMethodError: io.netty.channel.DefaultFileRegion.<init>(Ljava/io/File;JJ)V
Exclude netty jar from assembly 

#Solution - copy latest ie spark one to hadoop 
#Hadoop has nettry jar at 
./common/lib/netty-3.6.2.Final.jar
./hdfs/lib/netty-3.6.2.Final.jar
./mapreduce/lib/netty-3.6.2.Final.jar
./tools/lib/netty-3.6.2.Final.jar
./yarn/lib/netty-3.6.2.Final.jar
./httpfs/tomcat/webapps/webhdfs/WEB-INF/lib/netty-3.6.2.Final.jar
./httpfs/tomcat/webapps/webhdfs/WEB-INF/lib/netty-all-4.0.23.Final.jar
./kms/tomcat/webapps/kms/WEB-INF/lib/netty-3.6.2.Final.jar

#important one is 
./hdfs/lib/netty-all-4.0.23.Final.jar


#copy 
$ cd c:\hadoop
$ rm ./hdfs/lib/netty-all-4.0.23.Final.jar
$ cp c:\spark\jars\netty-all-4.0.42.Final.jar ./hdfs/lib/




###Spark - Difference between ML and Mlib 
#ML - recomended to use this 
•New
•Pipelines
•Dataframes
•Easier to construct a practical machine learning pipeline

#MLlib
•Old
•RDD
•More features
  
  
  
/***** Machine Learning Data Types(ML/MLIB) ****/
###Spark - ML and MLIB Data types 
#dense vectors:
•NumPy's array
•Python's list, e.g., [1, 2, 3]
#sparse vectors:
•MLlib's SparseVector.
•SciPy's csc_matrix with a single column

#For ML - uses Dataframe (pyspark.sql module)
#Hence LocalMatrix, DistributedMatrix etc do not have significance 


#Use Vectors to create sparse vector 
#http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.Vectors
#for anaconda2 as it has numpy installed 

##Spark- MLib datatypes 

#commonMethods- Each below class has 
.toArray()
    Returns an numpy.ndarray    
.asML()
    to ML equivalent type 
    
#Note 
#Numpy array, list, SparseVector, or SciPy sparse etc are interchangable
#can be used in any args  
pyspark.mllib.linalg module
    Vector
    Vectors #Factory methods
        static dense(*elements)
            Vectors.dense([1.0, 2.0])
        static norm(vector, p)
        static fromML(vec)
        static parse(s)
            >>> Vectors.parse('[2,1,2 ]')
            DenseVector([2.0, 1.0, 2.0])
            >>> Vectors.parse(' ( 100,  [0],  [2])')
            SparseVector(100, {0: 2.0})
        static sparse(size, *args)
                •args – Non-zero entries, as a dictionary, 
                        list of tuples, or two sorted lists 
                        containing indices and values.
            >>> Vectors.sparse(4, {1: 1.0, 3: 5.5})
            SparseVector(4, {1: 1.0, 3: 5.5})
            >>> Vectors.sparse(4, [(1, 1.0), (3, 5.5)])
            SparseVector(4, {1: 1.0, 3: 5.5})
            >>> Vectors.sparse(4, [1, 3], [1.0, 5.5])
            SparseVector(4, {1: 1.0, 3: 5.5})
        static squared_distance(v1, v2)
        static stringify(vector)
        static zeros(size)
    DenseVector(ar)
        Column-major dense matrix.
        +,-,*,/ with another DenseVector(elementwise)
            v = Vectors.dense([1.0, 2.0])
            u = Vectors.dense([3.0, 4.0])
            u + v
        +,-,/,* with int or float (elementwise)
        .methods(args)
            any ndarray instance methods 
        [index or slice]
            accessing , Note these are immutable  
        np.methods(args)
            numpy.methods() 
        dot(other)
        norm(p)
        numNonzeros()
        static parse(s)
            DenseVector.parse(' [ 0.0,1.0,2.0,  3.0]')
        squared_distance(other)
        values
    SparseVector(size, *args)
        Users may alternatively pass SciPy's {scipy.sparse} data types.
        •args – Non-zero entries, as a dictionary, 
                        list of tuples, or two sorted lists 
                        containing indices and values.
        dot(other)
            a = SparseVector(4, [1, 3], [3.0, 4.0])
            >>> a.dot(a)
            25.0
        norm(p)
        numNonzeros()
        static parse(s)
            SparseVector.parse(' (4, [0,1 ],[ 4.0,5.0] )')
        squared_distance(other)
        values
        indices
    Matrix(numRows, numCols, isTransposed=False)
    Matrices #factory 
        static dense(numRows, numCols, values)
        static sparse(numRows, numCols, colPtrs, rowIndices, values)
        static fromML(mat)
    DenseMatrix(numRows, numCols, values, isTransposed=False)
        toSparse()
    SparseMatrix(numRows, numCols, colPtrs, rowIndices, values, isTransposed=False)
        toDense()
    QRDecomposition(Q, R)
        Q
        R

##Spark- ML  datatypes 
#Numpy array, list, SparseVector, or SciPy sparse etc are interchangable
#can be used in any args  

#Note: args details follows Mlib args pattern, check there 

#commonMethods- Each below class has 
.toArray()
    Returns an numpy.ndarray
#No conversion to mllib types 

pyspark.ml.linalg module
    Vector
    Vectors #Factory methods
        static dense(*elements)
        static norm(vector, p)        
        static sparse(size, *args)
            •args – Non-zero entries, as a dictionary, list of tuples, or two sorted lists containing indices and values.
            >>> Vectors.sparse(4, {1: 1.0, 3: 5.5})
            SparseVector(4, {1: 1.0, 3: 5.5})
            >>> Vectors.sparse(4, [(1, 1.0), (3, 5.5)])
            SparseVector(4, {1: 1.0, 3: 5.5})
            >>> Vectors.sparse(4, [1, 3], [1.0, 5.5])
            SparseVector(4, {1: 1.0, 3: 5.5})
        static squared_distance(v1, v2)
        static stringify(vector)
        static zeros(size)
    DenseVector(ar)
        Column-major dense matrix.
        +,-,*,/ with another DenseVector(elementwise)
            v = Vectors.dense([1.0, 2.0])
            u = Vectors.dense([3.0, 4.0])
            u + v
        +,-,/,* with int or float (elementwise)
        .methods(args)
            any ndarray instance methods 
        [index or slice]
            accessing , Note these are immutable  
        np.methods(args)
            numpy.methods() 
        dot(other)
        norm(p)
        numNonzeros()       
        squared_distance(other)
        values
    SparseVector(size, *args)
        dot(other)
        norm(p)
        numNonzeros()        
        squared_distance(other)
        values
        indices
    Matrix(numRows, numCols, isTransposed=False)
    DenseMatrix(numRows, numCols, values, isTransposed=False)
            m = DenseMatrix(2, 2, range(4))
            >>> m.toArray()
                array([[ 0.,  2.],
                       [ 1.,  3.]])
        toSparse()
    SparseMatrix(numRows, numCols, colPtrs, rowIndices, values, isTransposed=False)
        toDense()
    Matrices #factory 
        static dense(numRows, numCols, values)
        static sparse(numRows, numCols, colPtrs, rowIndices, values)
      
    

 



##Spark - Example 
import numpy as np
from pyspark.mllib.linalg import Vectors, DenseVector, SparseVector

# Use a NumPy array as a dense vector.
dv1 = np.array([1.0, 0.0, 3.0])
# Use a Python list as a dense vector.
dv2 = [1.0, 0.0, 3.0]
# Create a SparseVector.
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])  # first arg is size, then index, value 
#Operations
>>> v = Vectors.dense([1.0, 2.0])
>>> u = Vectors.dense([3.0, 4.0])
>>> v + u
DenseVector([4.0, 6.0])
>>> 2 - v
DenseVector([1.0, 0.0])
>>> v / 2
DenseVector([0.5, 1.0])
>>> v * u
DenseVector([3.0, 8.0])
>>> u / v
DenseVector([3.0, 2.0])
>>> u % 2
DenseVector([1.0, 0.0])
>>> np.sqrt(u)
array([ 1.73205081,  2.        ])

#access 
u[0]
u[0:]   #gets np.array 

#parse from string 
>>> DenseVector.parse(' [ 0.0,1.0,2.0,  3.0]')
DenseVector([0.0, 1.0, 2.0, 3.0])
>>> SparseVector.parse(' (4, [0,1 ],[ 4.0,5.0] )')
SparseVector(4, {0: 4.0, 1: 5.0})

#Squared distance of two Vectors.
>>> dense.squared_distance(dense)
0.0

#ToSTring
>>> Vectors.stringify(Vectors.sparse(2, [1], [1.0]))
'(2,[1],[1.0])'
>>> Vectors.stringify(Vectors.dense([0.0, 1.0]))
'[0.0,1.0]'



###Spark - MLIB and ML - Local matrix - not backed by a RDD 
#A local matrix has integer-typed row and column indices and double-typed values, stored on a single machine
#DenseMatrix - [1.0, 3.0, 5.0, 2.0, 4.0, 6.0] with the matrix size (3, 2).(column-major order)

pyspark.mllib.linalg.DenseMatrix(numRows, numCols, values, isTransposed=False)
    Matrix entries in column major if not transposed 
    or in row major otherwise
    asML()
    toArray()
    toSparse()
pyspark.mllib.linalg.SparseMatrix(numRows, numCols, colPtrs, owIndices, values, isTransposed=False)
    Sparse Matrix stored in CSC format.  
    asML()
    toArray()
    toDense()
pyspark.mllib.linalg.Matrices
    static dense(numRows, numCols, values)
        Create a DenseMatrix
    static fromML(mat)
        This does NOT copy the data; it copies references.
    static sparse(numRows, numCols, colPtrs, rowIndices, values)
        Create a SparseMatrix
    
pyspark.ml.linalg.DenseMatrix(numRows, numCols, values, isTransposed=False)
    Matrix entries in column major if not transposed 
    or in row major otherwise
    toArray()
    toSparse()    
pyspark.ml.linalg.SparseMatrix(numRows, numCols, colPtrs, rowIndices, values, isTransposed=False)
    Sparse Matrix stored in CSC format.
    toArray()
    toDense()
pyspark.ml.linalg.Matrices
    static dense(numRows, numCols, values)
        Create a DenseMatrix
    static sparse(numRows, numCols, colPtrs, rowIndices, values)
        Create a SparseMatri    


#Sparse Matrix stored in Compressed Sparse Column (CSC) format in column-major order
#CSC is (val, col_ptr, row_ind), where val is an array of the (top-to-bottom, then left-to-right) non-zero values of the matrix; 
#row_ind is the row indices of 2D matrix corresponding to the values; 
#col_ptr is the list of indexes of 'val'  where each column starts
#Example in scipy 
import numpy as np
import scipy.sparse as sps
#for matrix 
[[1, 0, 2],
 [0, 0, 3],
 [4, 5, 6]]
             #index  0  1  2  3  4  5
>>> data = np.array([1, 4, 5, 2, 3, 6]) #(top-to-bottom, then left-to-right)
>>> indices = np.array([0, 2, 2, 0, 1, 2]) #row indices of 2D matrix traversing top-to-bottom, then left-to-right
>>> indptr = np.array([0, 2, 3, 6])  #index of indices where column starts , last one is len(indices)
>>> mtx = sps.csc_matrix((data, indices, indptr), shape=(3, 3))
>>> mtx.todense()
matrix([[1, 0, 2],
        [0, 0, 3],
        [4, 5, 6]])

        

##*PY 
#python Matrix and Matrices are trimmed down version of scala, check method before using 
#Use Numpy

from pyspark.mllib.linalg import Matrix, Matrices

# Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
dm2 = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])

# Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])


>>> dm2.toArray()  #numpy array 
array([[ 1.,  4.],
       [ 2.,  5.],
       [ 3.,  6.]])
>>> type(dm2.toArray())
<type 'numpy.ndarray'>
#all numpy methods can be used as given in Vector chapter 

##Quick - Numpy - Creation of array - these methods take (m,n) dimensions 

import numpy as np

np.zeros((2,2))  # Create an array of all zeros 
np.ones((1,2))   # Create an array of all ones
np.full((2,2), 7) # Create a constant array
np.eye(2)        # Create a 2x2 identity matrix
np.random.random((2,2)) # Create an array filled with random values
arange([start,] stop[, step,][, dtype]) 			Return evenly spaced values within a given interval.
linspace(start, stop[, num, endpoint, ...]) 		Return evenly spaced numbers over a specified interval.
logspace(start, stop[, num, endpoint, base, ...]) 	Return numbers spaced evenly on a log scale.





##Spark - MLIB -  Labeled point

class pyspark.mllib.regression.LabeledPoint(label, features)
    Class that represents the features and labels of a data point.
    •label – Label for this data point.
    •features – Vector of features for this point 
                (NumPy array, list, pyspark.mllib.linalg.SparseVector, or scipy.sparse column matrix).

#Example 
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

# Create a labeled point with a positive label and a dense feature vector.
pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])

# Create a labeled point with a negative label and a sparse feature vector.
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))



##Spark - MLIB -  Sparse data - LibSVM
# MLlib supports reading training examples stored in LIBSVM format, 
#which is the default format used by LIBSVM and LIBLINEAR
#text format as below , for feature1, index1 is index of sparse vecctor where value is value1 
label index1:value1 index2:value2 ...

#Example 
from pyspark.mllib.util import MLUtils
examples = MLUtils.loadLibSVMFile(sc, "../data/sample_libsvm_data.txt")

#for ML, use spark instance of SparkSession 
val dataFrame = spark.read.format("libsvm").load("../data/sample_libsvm_data.txt")





##Spark - MLIB -  mllib distributed - Backed by a RDD 
#for ML, use DataFrame 
pyspark.mllib.linalg.distributed module
    DistributedMatrix
        Base class all belocw classes
    BlockMatrix(blocks, rowsPerBlock, colsPerBlock, numRows=0, numCols=0)
        blocks - An RDD of sub-matrix blocks ((blockRowIndex, blockColIndex), sub-matrix) 
            add(other)
            subtract(other)
            blocks
            cache()
            multiply(other)
            transpose()
            colsPerBlock
            numColBlocks
            numCols()
            numRowBlocks
            numRows()
            persist(storageLevel)
            rowsPerBlock
            toCoordinateMatrix()
            toIndexedRowMatrix()
            toLocalMatrix()      #Only BlockMatrix has conversion to local Matrix, others dont have, But others can be converted to BlockMatrix
            validate()
    MatrixEntry(long i, long j, float val)    
    CoordinateMatrix(entries, numRows=0, numCols=0)
        entries – An RDD of MatrixEntry(long i, long j, float val) inputs 
        or (long, long, float) tuples
            entries  #RDD
                Entries of the CoordinateMatrix stored as an RDD of MatrixEntries.
            numCols()
            numRows()
            toBlockMatrix(rowsPerBlock=1024, colsPerBlock=1024)
            toIndexedRowMatrix()
            toRowMatrix()
            transpose()
    RowMatrix(rows, numRows=0, numCols=0)
        Represents a row-oriented distributed Matrix with no meaning of  row indices.
        •rows – An RDD of vectors., each value is for a feature 
            columnSimilarities(threshold=0.0)
                Compute similarities between columns of this matrix.                 
                Returns:An n x n sparse upper-triangular CoordinateMatrix of cosine similarities between columns of this matrix. 
                >>> rows = sc.parallelize([[1, 2], [1, 5]])
                >>> mat = RowMatrix(rows)
                >>> sims = mat.columnSimilarities()
                >>> sims.entries.first().value
                0.91914503...
            computeColumnSummaryStatistics()
                Computes column-wise summary statistics.
                Returns: MultivariateStatisticalSummary , 
                         has count()max()mean()min()normL1()normL2()numNonzeros()variance()                 
                >>> rows = sc.parallelize([[1, 2, 3], [4, 5, 6]])
                >>> mat = RowMatrix(rows)
                >>> colStats = mat.computeColumnSummaryStatistics()
                >>> colStats.mean()
                array([ 2.5,  3.5,  4.5])        
            computeCovariance()
                Computes the covariance matrix, treating each row as an observation
                >>> rows = sc.parallelize([[1, 2], [2, 1]])
                >>> mat = RowMatrix(rows)
                >>> mat.computeCovariance()
                DenseMatrix(2, 2, [0.5, -0.5, -0.5, 0.5], 0)
            computeGramianMatrix()
                Computes the Gramian matrix A^T A.                
                >>> rows = sc.parallelize([[1, 2, 3], [4, 5, 6]])
                >>> mat = RowMatrix(rows)
                >>> mat.computeGramianMatrix()
                DenseMatrix(3, 3, [17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0], 0)
            computePrincipalComponents(k)
                Computes the k principal components of the given row matrix
                Returns:  pyspark.mllib.linalg.DenseMatrix 
                >>> rows = sc.parallelize([[1, 2, 3], [2, 4, 5], [3, 6, 1]])
                >>> rm = RowMatrix(rows)
                >>> # Returns the two principal components of rm
                >>> pca = rm.computePrincipalComponents(2)
                >>> pca
                DenseMatrix(3, 2, [-0.349, -0.6981, 0.6252, -0.2796, -0.5592, -0.7805], 0)
                >>> # Transform into new dimensions with the greatest variance.
                >>> rm.multiply(pca).rows.collect() 
                [DenseVector([0.1305, -3.7394]), DenseVector([-0.3642, -6.6983]),         DenseVector([-4.6102, -4.9745])]
            computeSVD(k, computeU=False, rCond=1e-09)
                Computes the singular value decomposition of the RowMatrix.
                Returns:SingularValueDecomposition(U,V,s)
                    •U: (m X k) (left singular vectors) is a RowMatrix whose
                    columns are the eigenvectors of (A X A')
                    •s: DenseVector consisting of square root of the eigenvalues
                    (singular values) in descending order.
                    •v: (n X k) (right singular vectors) is a Matrix whose columns
                    are the eigenvectors of (A' X A)
                >>> rows = sc.parallelize([[3, 1, 1], [-1, 3, 1]])
                >>> rm = RowMatrix(rows)
                >>> svd_model = rm.computeSVD(2, True)
                >>> svd_model.U.rows.collect()
                [DenseVector([-0.7071, 0.7071]), DenseVector([-0.7071, -0.7071])]
                >>> svd_model.s
                DenseVector([3.4641, 3.1623])
                >>> svd_model.V
                DenseMatrix(3, 2, [-0.4082, -0.8165, -0.4082, 0.8944, -0.4472, 0.0], 0)
            multiply(matrix)
            numCols()
            numRows()
            rows
                Rows of the RowMatrix stored as an RDD of vectors.
                >>> mat = RowMatrix(sc.parallelize([[1, 2, 3], [4, 5, 6]]))
                >>> rows = mat.rows  #RDD
                >>> rows.first()
                DenseVector([1.0, 2.0, 3.0])
            tallSkinnyQR(computeQ=False)
                Compute the QR decomposition of this RowMatrix.
                Returns: QRDecomposition(Q: RowMatrix, R: Matrix), where Q = None if computeQ = false. 
                >>> rows = sc.parallelize([[3, -6], [4, -8], [0, 1]])
                >>> mat = RowMatrix(rows)
                >>> decomp = mat.tallSkinnyQR(True)
                >>> Q = decomp.Q
                >>> R = decomp.R
                >>> # Test with absolute values
                >>> absQRows = Q.rows.map(lambda row: abs(row.toArray()).tolist())
                >>> absQRows.collect()
                [[0.6..., 0.0], [0.8..., 0.0], [0.0, 1.0]]
                >>> # Test with absolute values
                >>> abs(R.toArray()).tolist()
                [[5.0, 10.0], [0.0, 1.0]]
    IndexedRow(index, vector)
    IndexedRowMatrix(rows, numRows=0, numCols=0)
        rows – An RDD of IndexedRow(index, vector_for_that_row) or (long, vector) tuples.
            numCols()
            numRows()            
            multiply(matrix)
            computeGramianMatrix()
                Computes the Gramian matrix A^T A.
            columnSimilarities()
                Compute all cosine similarities between columns.
                >>> rows = sc.parallelize([IndexedRow(0, [1, 2, 3]),IndexedRow(6, [4, 5, 6])])
                >>> mat = IndexedRowMatrix(rows)
                >>> cs = mat.columnSimilarities()
                >>> print(cs.numCols())
                3
            computeSVD(k, computeU=False, rCond=1e-09)     
                Returns: SingularValueDecomposition object 
                >>> rows = [(0, (3, 1, 1)), (1, (-1, 3, 1))]
                >>> irm = IndexedRowMatrix(sc.parallelize(rows))
                >>> svd_model = irm.computeSVD(2, True)
                >>> svd_model.U.rows.collect() 
                [IndexedRow(0, [-0.707106781187,0.707106781187]), IndexedRow(1, [-0.707106781187,-0.707106781187])]
                >>> svd_model.s
                DenseVector([3.4641, 3.1623])
                >>> svd_model.V
                DenseMatrix(3, 2, [-0.4082, -0.8165, -0.4082, 0.8944, -0.4472, 0.0], 0)               
            rows
                Rows of the IndexedRowMatrix stored as an RDD of IndexedRows.
                >>> mat = IndexedRowMatrix(sc.parallelize([IndexedRow(0, [1, 2, 3]),IndexedRow(1, [4, 5, 6])]))
                >>> rows = mat.rows #RDD
                >>> rows.first()
                IndexedRow(0, [1.0,2.0,3.0])
                >>> mat.toBlockMatrix().toLocalMatrix()
                DenseMatrix(2, 3, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 0)
                >>> mat.toBlockMatrix().toLocalMatrix().toArray()
                array([[1., 2., 3.],
                       [4., 5., 6.]])
            toBlockMatrix(rowsPerBlock=1024, colsPerBlock=1024)    
            toCoordinateMatrix()    
            toRowMatrix()












##Spark - MLIB - Distributed matrix in details 
#A distributed matrix has long-typed row and column indices and double-typed values, stored distributively in one or more RDDs
#Converting a distributed matrix to a different format may require a global shuffle, which is quite expensive


##Spark - MLIB - Distributed matrix- RowMatrix
#A RowMatrix is a row-oriented distributed matrix without meaningful row indices, 
#backed by an RDD of its rows, where each row is a local vector.

from pyspark.mllib.linalg.distributed import RowMatrix

# Create an RDD of vectors.
rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
>>> rows.first()
[1, 2, 3]

# Create a RowMatrix from an RDD of vectors.
mat = RowMatrix(rows)


# Get its size.
m = mat.numRows()  # 4
n = mat.numCols()  # 3

# Get the rows as an RDD of vectors again.
rowsRDD = mat.rows

 
>>> colStats = mat.computeColumnSummaryStatistics()
>>> colStats.mean()
array([ 2.5,  3.5,  4.5])

>>> mat.computeCovariance()
>>> mat.computeGramianMatrix()
#QR
>>> decomp = mat.tallSkinnyQR(True)
>>> Q = decomp.Q
>>> R = decomp.R
import numpy as np 
>>> # Test with absolute values
>>> absQRows = Q.rows.map(lambda row: np.abs(row.toArray()).tolist())
>>> absQRows.collect()
[[0.6..., 0.0], [0.8..., 0.0], [0.0, 1.0]]

>>> # Test with absolute values
>>> np.abs(R.toArray()).tolist()
[[5.0, 10.0], [0.0, 1.0]]



##Spark - MLIB - Distributed matrix- IndexedRowMatrix 
#similar to a RowMatrix but with meaningful row indices. 
#It is backed by an RDD of indexed rows, so that each row is represented by its index (long-typed) and a local vector.
#RowMatrix methods are applicable here as well 


from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix

# Create an RDD of indexed rows.
#   - This can be done explicitly with the IndexedRow class:
indexedRows = sc.parallelize([IndexedRow(0, [1, 2, 3]),
                              IndexedRow(1, [4, 5, 6]),
                              IndexedRow(2, [7, 8, 9]),
                              IndexedRow(3, [10, 11, 12])])
#   - or by using (long, vector) tuples:
indexedRows = sc.parallelize([(0, [1, 2, 3]), (1, [4, 5, 6]),
                              (2, [7, 8, 9]), (3, [10, 11, 12])])

# Create an IndexedRowMatrix from an RDD of IndexedRows.
mat = IndexedRowMatrix(indexedRows)

# Get its size.
m = mat.numRows()  # 4
n = mat.numCols()  # 3

# Get the rows as an RDD of IndexedRows.
rowsRDD = mat.rows
rowsRDD.collect()

# Convert to a RowMatrix by dropping the row indices.
rowMat = mat.toRowMatrix()

>>> cs = mat.columnSimilarities()
>>> print(cs.numCols())
3
>>> mat.computeGramianMatrix()
DenseMatrix(3, 3, [17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0], 0)
#with Block Matrix 
>>> rows = sc.parallelize([IndexedRow(0, [1, 2, 3]),
                       IndexedRow(6, [4, 5, 6])])
>>> mat = IndexedRowMatrix(rows).toBlockMatrix()

>>> # This IndexedRowMatrix will have 7 effective rows, due to
>>> # the highest row index being 6, and the ensuing
>>> # BlockMatrix will have 7 rows as well.
>>> print(mat.numRows())
7
>>> print(mat.numCols())
3
>>> mat.toLocalMatrix()




##Spark - MLIB - CoordinateMatrix
#A CoordinateMatrix is a distributed matrix backed by an RDD of its entries
    MatrixEntry(i, j, value)
#Each entry is a tuple of (i: Long, j: Long, value: Double), where i is the row index, j is the column index, 
#and value is the entry value


from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

# Create an RDD of coordinate entries.
#   - This can be done explicitly with the MatrixEntry class:
entries = sc.parallelize([MatrixEntry(0, 0, 1.2), MatrixEntry(1, 0, 2.1), MatrixEntry(6, 1, 3.7)])
#   - or using (long, long, float) tuples:
entries = sc.parallelize([(0, 0, 1.2), (1, 0, 2.1), (2, 1, 3.7)])

# Create an CoordinateMatrix from an RDD of MatrixEntries.
mat = CoordinateMatrix(entries)

# Get its size.
m = mat.numRows()  # 3
n = mat.numCols()  # 2

# Get the entries as an RDD of MatrixEntries.
entriesRDD = mat.entries
entriesRDD.collect() 

# Convert to a RowMatrix.
rowMat = mat.toRowMatrix()

# Convert to an IndexedRowMatrix.
indexedRowMat = mat.toIndexedRowMatrix()

# Convert to a BlockMatrix.
blockMat = mat.toBlockMatrix()

>>> mat_transposed = mat.transpose()





##Spark - MLIB - BlockMatrix
BlockMatrix(blocks, rowsPerBlock, colsPerBlock, numRows=0, numCols=0)
#A BlockMatrix is a distributed matrix backed by an RDD of blocks 
#where a block = tuple of ((Int, Int), Matrix), 
#where the (Int, Int) is the index of the block, 
#and Matrix is the sub-matrix at the given index with size rowsPerBlock x colsPerBlock.



from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix

# Create an RDD of sub-matrix blocks.
blocks = sc.parallelize([((0, 0), Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])),
                         ((1, 0), Matrices.dense(3, 2, [7, 8, 9, 10, 11, 12]))])

# Create a BlockMatrix from an RDD of sub-matrix blocks.
mat = BlockMatrix(blocks, 3, 2)

# Get its size.
m = mat.numRows()  # 6
n = mat.numCols()  # 2

# Get the blocks as an RDD of sub-matrix blocks.
blocksRDD = mat.blocks
blocksRDD.collect()

# Convert to a LocalMatrix.
localMat = mat.toLocalMatrix()

# Convert to an IndexedRowMatrix.
indexedRowMat = mat.toIndexedRowMatrix()

# Convert to a CoordinateMatrix.
coordinateMat = mat.toCoordinateMatrix()

#Validate whether the BlockMatrix is set up properly. Throws an Exception when it is not valid.
#Nothing happens if it is valid.
mat.validate()

#Calculate A^T A.
ata = mat.transpose().multiply(mat)
ata.toLocalMatrix() #DenseMatrix(numRows: Int, numCols: Int, values: Array[Double], isTransposed: Boolean)
#matrix entries in column major if not transposed or in row major otherwise
ata2 = mat.add(mat)
ata2.toLocalMatrix()




























/***** RDD ****/
###Spark - RDD - Basic operations 
#every Spark application consists of a driver program that runs the user's main function (driver)
#and executes various parallel operations on a cluster(called executors on workers)


###Resilient distributed dataset (RDD) (think of Vector )
#collection/Vector of elements/objects (eg String, Vector, Matrix) 
#partitioned across the nodes of the cluster that can be operated on in parallel. 


###Shared variables(broadcast and accumulators) 
#By default, when Spark runs a function in parallel as a set of tasks on different nodes, 
#it ships a copy of each variable used in the function to each task
#Hence direct sharing of global variables not possible 

broadcast variables : Immutable global like varaible , to cache a value in memory on all nodes, 
accumulators        : Mutable shared variables that are only 'added' to, such as counters and sums.



#Only one SparkContext may be active per JVM. 
#You must stop() the active SparkContext before creating a new one.

from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
#use and then stop 
sc.stop()


##master is a Spark, Mesos or YARN cluster URL, or a special 'local' string to run in local mode. 
#Note, standalone code must call setMaster(master), but via spark-submit , one passes via command line 
#Note spark-shell, pyspark, sparkR, spark-submit all share same command line args 

$ pyspark --master local[4]

# to also add code.py to the search path (in order to later be able to import code), use:
$ pyspark --master local[4] --py-files code.py



##Spark - RDD - sparkContext 
# http://spark.apache.org/docs/2.1.0/api/python/pyspark.html#pyspark.SparkContext

class pyspark.SparkConf(loadDefaults=True, _jvm=None, _jconf=None)
    Used to set various Spark parameters as key-value pairs.
    https://spark.apache.org/docs/latest/configuration.html
    All setter methods in this class support chaining. 
    eg conf.setMaster('local').setAppName('My app').
        contains(key)
        get(key, defaultValue=None)
        getAll()
        set(key, value)
        setAll(pairs)
        setAppName(value)
        setExecutorEnv(key=None, value=None, pairs=None)
            Set an environment variable to be passed to executors.
        setIfMissing(key, value)
            Set a configuration property, if not already set.
        setMaster(value)
        setSparkHome(value)
        toDebugString()

class pyspark.SparkContext(master=None, appName=None, sparkHome=None, 
            pyFiles=None, environment=None, batchSize=0, 
            serializer=PickleSerializer(), conf=None, 
            gateway=None, jsc=None, profiler_cls=<class 'pyspark.profiler.BasicProfiler'>)
    Main entry point for Spark functionality. 
        conf = SparkConf().setAppName(appName).setMaster(master)
        sc = SparkContext(conf=conf)
    Note path=local(available on all nodes), hdfs,or an HTTP, HTTPS or FTP URI
    PACKAGE_EXTENSIONS = ('.zip', '.egg', '.jar')
        accumulator(value, accum_param=None)
        broadcast(value)
        parallelize(c, numSlices=None)
            Distribute a local Python collection to form an RDD. 
            Using xrange(py2.7)/range(py3) is recommended 
            >>> sc.parallelize([0, 2, 3, 4, 6], 5).glom().collect()
            [[0], [2], [3], [4], [6]]
            >>> sc.parallelize(xrange(0, 6, 2), 5).glom().collect()
            [[], [0], [], [2], [4]]
        range(start, end=None, step=1, numSlices=None)
            Create a new RDD of int containing elements from start to end (exclusive), 
            >>> sc.range(5).collect()
            [0, 1, 2, 3, 4]
            >>> sc.range(2, 4).collect()
            [2, 3]
            >>> sc.range(1, 7, 2).collect()
            [1, 3, 5]
        emptyRDD()
            Create an RDD that has no partitions or elements.
        addFile(path, recursive=False)
            Add a file(path=local, hdfs,or an HTTP, HTTPS or FTP URI) 
            to be downloaded with this Spark job on every node. 
                >>> from pyspark import SparkFiles
                >>> path = os.path.join(tempdir, "test.txt")
                >>> with open(path, "w") as testFile:
                        _ = testFile.write("100")
                >>> sc.addFile(path)
                >>> def func(iterator):
                        with open(SparkFiles.get("test.txt")) as testFile:
                            fileVal = int(testFile.readline())
                        return [x * fileVal for x in iterator]
                >>> sc.parallelize([1, 2, 3, 4]).mapPartitions(func).collect()
                [100, 200, 300, 400]
        addPyFile(path), path=local, hdfs,or an HTTP, HTTPS or FTP URI
            Add a .py or .zip dependency for all tasks         
        binaryRecords(path, recordLength)
        binaryFiles(path, minPartitions=None)
            Read a directory of binary files from HDFS, local (available on all nodes), 
            or any Hadoop-supported file system URI    as a byte array. 
            Each file is read as a single record and returned in a (file,content) key-value pair
        hadoopFile(path, inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)
        hadoopRDD(inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)
        newAPIHadoopFile(path, inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)
        newAPIHadoopRDD(inputFormatClass, keyClass, valueClass, keyConverter=None, valueConverter=None, conf=None, batchSize=0)
        sequenceFile(path, keyClass=None, valueClass=None, keyConverter=None, valueConverter=None, minSplits=None, batchSize=0)
            Read a Hadoop SequenceFile(=(K,V) pair) with arbitrary key and value Writable class from HDFS, a local file system (available on all nodes), 
            or any Hadoop-supported file system URI
            A Hadoop configuration can be passed in as a Python dict. 
            •path – path to Hadoop file
            •inputFormatClass – fully qualified classname of Hadoop InputFormat (e.g. 'org.apache.hadoop.mapreduce.lib.input.TextInputFormat')
            •keyClass – fully qualified classname of key Writable class (e.g. 'org.apache.hadoop.io.Text')
            •valueClass – fully qualified classname of value Writable class (e.g. 'org.apache.hadoop.io.LongWritable')
            •keyConverter – (None by default)
            •valueConverter – (None by default)
            •conf – Hadoop configuration, passed in as a dict (None by default)
            •batchSize – The number of Python objects represented as a single Java object. 
                         (default 0, choose batchSize automatically)
            #keyClass or valueClass
            org.apache.hadoop.io.IntWritable
            org.apache.hadoop.io.ByteWritable
            org.apache.hadoop.io.DoubleWritable
            org.apache.hadoop.io.FloatWritable
            org.apache.hadoop.io.IntWritable
            org.apache.hadoop.io.LongWritable
            org.apache.hadoop.io.Text
            #Create a RDD of K,V and save as sequence file 
            >>> sc.parallelize([("foo", '{"foo": 1}'), ("bar", '{"bar": 2}')]).saveAsSequenceFile("example")
            #Read K,V 
            rdd_k_v = sc.sequenceFile("example", "org.apache.hadoop.io.Text", "org.apache.hadoop.io.Text")
            >>> rdd_k_v.collect()
            [('foo', '{"foo": 1}'), ('bar', '{"bar": 2}')]
            rdd_v = rdd_k_v.values()  
            >>> rdd_v.first()
            '{"foo": 1}'            
        pickleFile(name, minPartitions=None)
            Load an RDD previously saved using RDD.saveAsPickleFile method.
            >>> tmpFile = tempfile.NamedTemporaryFile(delete=True)
            >>> tmpFile.close()
            >>> sc.parallelize(range(10)).saveAsPickleFile(tmpFile.name, 5)
            >>> sorted(sc.pickleFile(tmpFile.name, 3).collect())
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        textFile(name, minPartitions=None, use_unicode=True)
            Read a text file linewise 
            If use_unicode is False, the strings will be kept as str (encoding as utf-8)
            >>> path = os.path.join(tempdir, "sample-text.txt")
            >>> with open(path, "w") as testFile:
                    _ = testFile.write("Hello world!")
            >>> textFile = sc.textFile(path)
            >>> textFile.collect()
            [u'Hello world!']
        wholeTextFiles(path, minPartitions=None, use_unicode=True)
            Read a directory of text files as (file,content) pair
            #Example 
            hdfs://a-hdfs-path/part-00000
            hdfs://a-hdfs-path/part-00001
            ...
            hdfs://a-hdfs-path/part-nnnnn

            rdd = sparkContext.wholeTextFiles('hdfs://a-hdfs-path')
            #then rdd contains:
            (a-hdfs-path/part-00000, its content)
            (a-hdfs-path/part-00001, its content)
            ...
            (a-hdfs-path/part-nnnnn, its content)
        union(rdds)
            Build the union of a list of RDDs.
            >>> path = os.path.join(tempdir, "union-text.txt")
            >>> with open(path, "w") as testFile:
                    _ = testFile.write("Hello")
            >>> textFile = sc.textFile(path)
            >>> textFile.collect()
            [u'Hello']
            >>> parallelized = sc.parallelize(["World!"])
            >>> sorted(sc.union([textFile, parallelized]).collect())
            [u'Hello', 'World!']
        runJob(rdd, partitionFunc, partitions=None, allowLocal=False)
            Executes the given partitionFunc on the specified set of partitions, 
            returning the result as an array of elements.
            >>> myRDD = sc.parallelize(range(6), 3)
            >>> sc.runJob(myRDD, lambda part: [x * x for x in part])
            [0, 1, 4, 9, 16, 25]
            >>> myRDD = sc.parallelize(range(6), 3)
            >>> sc.runJob(myRDD, lambda part: [x * x for x in part], [0, 2], True)
            [0, 1, 16, 25]          
        setJobGroup(groupId, description, interruptOnCancel=False)
            Assigns a group ID to all the jobs started by this thread 
            use SparkContext.cancelJobGroup to cancel all running jobs in this group.
            >>> import threading
            >>> from time import sleep
            >>> result = "Not Set"
            >>> lock = threading.Lock()
            >>> def map_func(x):
                    sleep(100)
                    raise Exception("Task should have been cancelled")
            >>> def start_job(x):
                    global result
                    try:
                        sc.setJobGroup("job_to_cancel", "some description")
                        result = sc.parallelize(range(x)).map(map_func).collect()
                    except Exception as e:
                        result = "Cancelled"
                    lock.release()
            >>> def stop_job():
                    sleep(5)
                    sc.cancelJobGroup("job_to_cancel")
            >>> supress = lock.acquire()
            >>> supress = threading.Thread(target=start_job, args=(10,)).start()
            >>> supress = threading.Thread(target=stop_job).start()
            >>> supress = lock.acquire()
            >>> print(result)
            Cancelled
        cancelAllJobs()
        cancelJobGroup(groupId)
        defaultMinPartitions
        defaultParallelism
        stop()
            Shut down the SparkContext.
        applicationId
            A unique identifier for the Spark application
            >>> sc.applicationId  
            u'local-...'
        setCheckpointDir(dirName)
            The directory must be a HDFS path if running on a cluster.  
        version
            The version of Spark on which this application is running.
        uiWebUrl
            Return the URL of the SparkUI instance started by this SparkContext
        setLocalProperty(key, value)
        setLogLevel(logLevel)
        getConf()
        getLocalProperty(key)
        classmethod getOrCreate(conf=None)
            Get or instantiate a SparkContext and register it as a singleton object.
        classmethod setSystemProperty(key, value)
            Set a Java system property, such as spark.executor.memory. 
            This must must be invoked before instantiating SparkContext.
        show_profiles()
        dump_profiles(path)  
        sparkUser()
        startTime
        statusTracker()           
            
            
class pyspark.SparkFiles
    Resolves paths to files added through SparkContext.addFile()
        classmethod get(filename)
            Get the absolute path of a file 
        classmethod getRootDirectory()
            Get the root directory that contains files added 

class pyspark.StorageLevel(useDisk, useMemory, useOffHeap, deserialized, replication=1)
    DISK_ONLY = StorageLevel(True, False, False, False, 1)
    DISK_ONLY_2 = StorageLevel(True, False, False, False, 2)
    MEMORY_AND_DISK = StorageLevel(True, True, False, False, 1)
    MEMORY_AND_DISK_2 = StorageLevel(True, True, False, False, 2)
    MEMORY_AND_DISK_SER = StorageLevel(True, True, False, False, 1)
    MEMORY_AND_DISK_SER_2 = StorageLevel(True, True, False, False, 2)
    MEMORY_ONLY = StorageLevel(False, True, False, False, 1)
    MEMORY_ONLY_2 = StorageLevel(False, True, False, False, 2)
    MEMORY_ONLY_SER = StorageLevel(False, True, False, False, 1)
    MEMORY_ONLY_SER_2 = StorageLevel(False, True, False, False, 2)
    OFF_HEAP = StorageLevel(True, True, True, False, 1)



#Some notes on reading files with Spark:
•If using a path on the local filesystem, 
 the file must also be accessible at the same path on worker nodes. 
 Either copy the file to all workers or use a network-mounted shared file system 

•All of Spark's file-based input methods, including textFile, 
 support running on directories, compressed files, and wildcards 
 textFile("/my/directory"), 
 textFile("/my/directory/*.txt")
 textFile("/my/directory/*.gz")

•SparkContext.wholeTextFiles lets you read a directory 
 containing multiple text files, 
 and returns each of them as (filename, content) pairs.
 This is in contrast with textFile, 
 which would return linewise in each file of the directory
 
•RDD.saveAsPickleFile(file) and SparkContext.pickleFile(file)
 support saving an RDD in pickle file 

•For SequenceFiles, use SparkContext's sequenceFile[K, V] method 
 where K and V are the types of key and values in the file. 
 These should be subclasses of Hadoop's Writable interface, 
 like IntWritable and Text. 
 

•For other Hadoop InputFormats, use the SparkContext.hadoopRDD method, 
 which takes an arbitrary JobConf 
 and input format class, key class and value class.
 Set these the same way you would for a Hadoop job with your input source. 
 Use SparkContext.newAPIHadoopRDD for InputFormats based on the 'new' MapReduce API (org.apache.hadoop.mapreduce).




##Spark - RDD - Resilient Distributed Datasets (RDDs) 

##Spark - RDD - creation by SparkContext 
#Typically you want 2-4 partitions for each CPU in your cluster. 
#create via parallelize or methods of reading textfile, json string etc 
#or indirectly via ML's DF read operations 

#RDD can not be printed, 
#must need to collect() at first... but may fail if data is large 

import pyspark 
import pyspark.rdd

data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)
distData.reduce(lambda a, b: a + b) 

#reading text file 
distFile = sc.textFile("d:/desktop/ppt/spark/data/README") #each line as one element in RDD 
distFile.map(lambda s: len(s)).reduce(lambda a, b: a + b)

#The key and value classes can be specified, 
#but for standard Writables this is not required.
>>> rdd = sc.parallelize(range(1, 4)).map(lambda x: (x, "a" * x)) #K,V 
>>> rdd.saveAsSequenceFile("d:/desktop/ppt/spark/data/file")              #hadoop sequence file, 
>>> sorted(sc.sequenceFile("d:/desktop/ppt/spark/data/file").collect(), key= lambda t: t[0] )
[(1, u'a'), (2, u'aa'), (3, u'aaa')]


##Spark - RDD - RDD Operations
#transformations, which create a new dataset from an existing one, 
#actions, which return a value to the driver program after running a computation on the dataset. 



lines = sc.textFile("d:/desktop/ppt/spark/data/README")
lineLengths = lines.map(lambda s: len(s))  #transformations
totalLength = lineLengths.reduce(lambda a, b: a + b) #action 
lineLengths.persist()


class pyspark.RDD(jrdd, ctx, jrdd_deserializer=AutoBatchedSerializer(PickleSerializer()))
    A Resilient Distributed Dataset (RDD), 
    the basic abstraction in Spark, can be operated on in parallel
    Many methods of RDD of (K,V) have partitionFunc=<function portable_hash>
    which is used for repartitioning by  partitionBy(numPartitions, partitionFunc)
    and partitionFunc takes only key 
    context
        The SparkContext that this RDD was created on.
    id()
        A unique ID for this RDD (within its SparkContext).
    collect()
        Return a list that contains all of the elements in this RDD.
        Must be called to get all data into driver program
    collectAsMap()
        For RDD of (K,V)
        Return the key-value pairs in this RDD to the master/driver as a dictionary.
        >>> m = sc.parallelize([(1, 2), (3, 4)]).collectAsMap()
        >>> m[1]
        2
        >>> m[3]
        4
    glom()
        Return an RDD by coalescing all elements within each partition into a list.
        >>> rdd = sc.parallelize([1, 2, 3, 4], 2)
        >>> sorted(rdd.glom().collect())
        [[1, 2], [3, 4]]
    aggregate(zeroValue, seqOp, combOp)
        Note for first seqOp, r=zeroValue, for 2nd..., r=return value of earlier seqOp
        seqOp(r,e): Aggregate the elements(e) of each partition, 
        combOp(r1,r2): Combine the results for all the partitions
        r and e type can be different
        op(r,x) can modify r and return , but not x 
            >>> seqOp = (lambda x, y: (x[0] + y, x[1] + 1)) #x,y is diff type . x is type of ZeroValue and also return type of function, y each element
            >>> combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))#x,y is of same type which is return of seqOp
            >>> sc.parallelize([1, 2, 3, 4]).aggregate((0, 0), seqOp, combOp)
            (10, 4)
            >>> sc.parallelize([]).aggregate((0, 0), seqOp, combOp)
            (0, 0)
    aggregateByKey(zeroValue, seqFunc, combFunc, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        Same as above, but for RDD of (K,V) pair 
        for Same key, K, seqFunc(V1,V2):R 
        combFunc(R1,R2) for merging values(of same Key) within same partition 
            >>> seqOp = lambda v1, v2: v1+v2  #ZeroValue is coming here 
            >>> combOp = lambda r1,r2: r1+r2 
            #sum by Key 
            >>> sc.parallelize([(0,1), (0,2), (1,3), (1,4)]).aggregateByKey(0, seqOp, combOp).collect()
            [(0, 3), (1, 7)]
            #Average by key 
            >>> seqOp = lambda t, v: (t[0]+v,t[1]+1)  #ZeroValue=(0,0), first index =sum, 2ndindex=count 
            >>> combOp = lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]) #simply add both     
            >>> sc.parallelize([(0,1), (0,2), (1,3), (1,4)]).aggregateByKey( (0,0) , seqOp, combOp).mapValues( lambda t : t[0]/t[1]).collect()
            [(0, 1.5), (1, 3.5)]            
    treeAggregate(zeroValue, seqOp, combOp, depth=2)
        Aggregates the elements of this RDD in a multi-level tree pattern.
        depth – suggested depth of the tree (default: 2) 
        >>> add = lambda x, y: x + y
        >>> rdd = sc.parallelize([-5, -4, -3, -2, -1, 1, 2, 3, 4], 10)
        >>> rdd.treeAggregate(0, add, add)
        -5
        >>> rdd.treeAggregate(0, add, add, 1)
        -5
        >>> rdd.treeAggregate(0, add, add, 2)
        -5
        >>> rdd.treeAggregate(0, add, add, 5)
        -5
        >>> rdd.treeAggregate(0, add, add, 10)
        -5
    combineByKey(createCombiner, mergeValue, mergeCombiners, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        For RDD of (K,V)
        Turns an RDD[(K, V)] into a result of type RDD[(K, C)], for a 'combined type' C.
        For same Key, K, 
            •createCombiner, which converts first V into a C 
            •mergeValue, to merge a V(from remaining Vs) into a C 
            •mergeCombiners, to combine two C's into a single one.
        >>> x = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> def add(a, b): return a + str(b)  #a type = return from createCombiner
        >>> sorted(x.combineByKey(str, add, add).collect())
        [('a', '11'), ('b', '1')]
    cogroup(other, numPartitions=None)
        For RDD of (K,V)
        For each key k in self or other, return a RDD of (k, ((v1,v11,...), (v2,v21..) ) )
        v1n are from self, v2n are from other 
        >>> x = sc.parallelize([("a", 1), ("b", 4)])
        >>> y = sc.parallelize([("a", 2)])
        #Note y= ((v1,v11,...), (v2,v21..)), actually (ResultIterable(v1n),ResultIterable(v2n))
        >>> [(x, tuple(map(list, y))) for x, y in sorted(list(x.cogroup(y).collect()))]
        [('a', ([1], [2])), ('b', ([4], []))]
    groupWith(other, *others)
        Alias for cogroup but with support for multiple RDDs.
        >>> w = sc.parallelize([("a", 5), ("b", 6)])
        >>> x = sc.parallelize([("a", 1), ("b", 4)])
        >>> y = sc.parallelize([("a", 2)])
        >>> z = sc.parallelize([("b", 42)])
        >>> [(x, tuple(map(list, y))) for x, y in sorted(list(w.groupWith(x, y, z).collect()))]
        [('a', ([5], [1], [2], [])), ('b', ([6], [4], [], [42]))]
    groupBy(f, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        Return an RDD of grouped items ie RDD of (K,V), where K=f(e) and V=[original_value1,...]
        >>> rdd = sc.parallelize([1, 1, 2, 3, 5, 8])
        >>> result = rdd.groupBy(lambda x: x % 2).collect()
        >>> sorted([(x, sorted(y)) for (x, y) in result])
        [(0, [2, 8]), (1, [1, 1, 3, 5])]
    groupByKey(numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        For RDD of (K,V)
        returns RDD of (K, (v1,v2..)) for same key 
        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.groupByKey().mapValues(len).collect())
        [('a', 2), ('b', 1)]
        >>> sorted(rdd.groupByKey().mapValues(list).collect())
        [('a', [1, 1]), ('b', [1])] 
    keyBy(f)
        Creates tuples (f(e), e) of the elements in this RDD by applying f(e).
        >>> x = sc.parallelize(range(0,3)).keyBy(lambda x: x*x)
        >>> x.collect()
        [(0, 0), (1, 1), (4, 2)]
    fullOuterJoin(other, numPartitions=None)
        For RDD of (K,V) (both self and other)
        Perform a right outer join of self and other.
        For each element (k, v) in self , 
        the resulting RDD will either contain all pairs (k, (v, w)) for w in other, 
        or the pair (k, (v, None)) if no elements in other have key k.
        Similarly for other 
        >>> x = sc.parallelize([("a", 1), ("b", 4)])
        >>> y = sc.parallelize([("a", 2), ("c", 8)])
        >>> sorted(x.fullOuterJoin(y).collect())
        [('a', (1, 2)), ('b', (4, None)), ('c', (None, 8))]
    join(other, numPartitions=None)
        For RDD of (K,V) 
        For each (k,v) of self, If there is same Key in other, returns  (k, (v1, v2)) tuple, 
        where (k, v1) is in self and (k, v2) is in other.
        >>> x = sc.parallelize([("a", 1), ("b", 4)])
        >>> y = sc.parallelize([("a", 2), ("a", 3)])
        >>> sorted(x.join(y).collect())
        [('a', (1, 2)), ('a', (1, 3))]   
    leftOuterJoin(other, numPartitions=None)
        For RDD of (K,V)
        For each element (k, v) in self, 
        returns RDD of pairs (k, (v, w)) for w in other, 
        or the pair (k, (v, None)) if no elements in other have key k.
        >>> x = sc.parallelize([("a", 1), ("b", 4)])
        >>> y = sc.parallelize([("a", 2)])
        >>> sorted(x.leftOuterJoin(y).collect())
        [('a', (1, 2)), ('b', (4, None))]
    rightOuterJoin(other, numPartitions=None)
        FOr RDD of (K,V)
        For each element (k, w) in other, 
        returns RDD of  pairs (k, (v, w)) for v in this, 
        or the pair (k, (None, w)) if no elements in self have key k.
        >>> x = sc.parallelize([("a", 1), ("b", 4)])
        >>> y = sc.parallelize([("a", 2)])
        >>> sorted(y.rightOuterJoin(x).collect())
        [('a', (2, 1)), ('b', (None, 4))]
    cache()
        Persist this RDD with the default storage level (MEMORY_ONLY).
    unpersist()
        Mark the RDD as non-persistent, 
        and remove all blocks for it from memory and disk.
    getStorageLevel()
        Get the RDD's current storage level.
        >>> rdd1 = sc.parallelize([1,2])
        >>> rdd1.getStorageLevel()
        StorageLevel(False, False, False, False, 1)
        >>> print(rdd1.getStorageLevel())
        Serialized 1x Replicated
    persist(storageLevel=StorageLevel(False, True, False, False, 1))
        This can only be used to assign a new storage level 
        if the RDD does not have a storage level set yet. 
        If no storage level is specified defaults to (MEMORY_ONLY).
        >>> rdd = sc.parallelize(["b", "a", "c"])
        >>> rdd.persist().is_cached
        True
    cartesian(other)
        Return RDD of all pairs of elements (a, b) where a is in self and b is in other.
        >>> rdd = sc.parallelize([1, 2])
        >>> sorted(rdd.cartesian(rdd).collect())
        [(1, 1), (1, 2), (2, 1), (2, 2)]
    coalesce(numPartitions, shuffle=False)
        Return a new RDD that is reduced into numPartitions partitions.
        >>> sc.parallelize([1, 2, 3, 4, 5], 3).glom().collect()
        [[1], [2, 3], [4, 5]]
        >>> sc.parallelize([1, 2, 3, 4, 5], 3).coalesce(1).glom().collect()
        [[1, 2, 3, 4, 5]]
    partitionBy(numPartitions, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        For RDD of (K,V), partitionFunc takes K and returns partition number where to put 
        Return a copy of the RDD partitioned using the specified partitioner.
        >>> pairs = sc.parallelize([1, 2, 3, 4, 2, 4, 1]).map(lambda x: (x, x))
        >>> pairs.partitionBy(2).glom().collect()
        [[(2, 2), (4, 4), (2, 2), (4, 4)], [(1, 1), (3, 3), (1, 1)]]
    repartition(numPartitions)
        Return a new RDD that has exactly numPartitions partitions.
        >>> rdd = sc.parallelize([1,2,3,4,5,6,7], 4)
        >>> sorted(rdd.glom().collect())
        [[1], [2, 3], [4, 5], [6, 7]]
        >>> len(rdd.repartition(2).glom().collect())
        2
        >>> len(rdd.repartition(10).glom().collect())
        10
    repartitionAndSortWithinPartitions(numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>, ascending=True, keyfunc=lambda x: x)
        For RDD of (K,V), repartition by partitionBy(numPartitions, partitionFunc)
        Within each resulting partition, sort records by their key with keyfunc(K) 
        >>> rdd = sc.parallelize([(0, 5), (3, 8), (2, 6), (0, 8), (3, 8), (1, 3)])
        #repartition with whether key is even or odd , note partitionFunc takes K 
        >>> rdd2 = rdd.repartitionAndSortWithinPartitions(2, lambda x: x % 2, True)
        >>> rdd2.glom().collect()
        [[(0, 5), (0, 8), (2, 6)], [(1, 3), (3, 8), (3, 8)]]
    first()
        Return the first element in this RDD.
        >>> sc.parallelize([2, 3, 4]).first()
        2
        >>> sc.parallelize([]).first()    
        ValueError: RDD is empty
    take(num)
        Take the first num elements of the RDD.(Action)
        >>> sc.parallelize([2, 3, 4, 5, 6]).cache().take(2)
        [2, 3]
        >>> sc.parallelize([2, 3, 4, 5, 6]).take(10)
        [2, 3, 4, 5, 6]
        >>> sc.parallelize(range(100), 100).filter(lambda x: x > 90).take(3)
        [91, 92, 93]
    takeOrdered(num, key=None)
        Get the N elements from an RDD ordered in ascending order 
        or as specified by the optional key(e) function.(Action)
        >>> sc.parallelize([10, 1, 2, 9, 3, 4, 5, 6, 7]).takeOrdered(6)
        [1, 2, 3, 4, 5, 6]
        >>> sc.parallelize([10, 1, 2, 9, 3, 4, 5, 6, 7], 2).takeOrdered(6, key=lambda x: -x)
        [10, 9, 7, 6, 5, 4]
    top(num, key=None)
        Get the top N elements from an RDD.(Action)
        >>> sc.parallelize([10, 4, 2, 12, 3]).top(1)
        [12]
        >>> sc.parallelize([2, 3, 4, 5, 6], 2).top(2)
        [6, 5]
        >>> sc.parallelize([10, 4, 2, 12, 3]).top(3, key=str)
        [4, 3, 2]
    count()
        Return the number of elements in this RDD.(Action)
        >>> sc.parallelize([2, 3, 4]).count()
        3
    countApprox(timeout, confidence=0.95)
        Approximate version of count() and returns result to master/driver(Action)
        >>> rdd = sc.parallelize(range(1000), 10)
        >>> rdd.countApprox(1000, 1.0)
        1000
    countApproxDistinct(relativeSD=0.05)
        Return approximate number of distinct elements in the RDD.
        >>> n = sc.parallelize(range(1000)).map(str).countApproxDistinct()
        >>> 900 < n < 1100
        True
        >>> n = sc.parallelize([i % 20 for i in range(1000)]).countApproxDistinct()
        >>> 16 < n < 24
        True
    countByKey()
        For RDD (K,V) and returns result (K,count) to master/driver(Action)
        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.countByKey().items())
        [('a', 2), ('b', 1)]
    countByValue()
        For only RDD of V, not for (K,V)
        Return the count of each unique value in this RDD 
        as a dictionary of (value, count) pairs.(Action)
        >>> sorted(sc.parallelize([1, 2, 1, 2, 2], 2).countByValue().items())
        [(1, 2), (2, 3)]
    distinct(numPartitions=None)
        Return a new RDD containing the distinct elements in this RDD.
        >>> sorted(sc.parallelize([1, 1, 2, 3]).distinct().collect())
        [1, 2, 3]
    min(key=None)
        Find the minimum item in this RDD.
    max(key=None)
        Find the maximum item in this RDD.
        key – A function used to generate key for comparing 
        >>> rdd = sc.parallelize([1.0, 5.0, 43.0, 10.0])
        >>> rdd.max()
        43.0
        >>> rdd.max(key=str)
        5.0
    mean()
        Compute the mean of this RDD's elements.
        >>> sc.parallelize([1, 2, 3]).mean()
        2.0
    meanApprox(timeout, confidence=0.95)
        Approximate operation to return the mean within a timeout or meet the confidence.
        >>> rdd = sc.parallelize(range(1000), 10)
        >>> r = sum(range(1000)) / 1000.0
        >>> abs(rdd.meanApprox(1000) - r) / r < 0.05
        True
    sampleStdev()
        Compute the sample standard deviation of this RDD's elements 
        >>> sc.parallelize([1, 2, 3]).sampleStdev()
        1.0
    sampleVariance()
        Compute the sample variance of this RDD's elements 
        >>> sc.parallelize([1, 2, 3]).sampleVariance()
        1.0
    stats()
        Return a StatCounter object containing below methods
        merge(value),mergeStats(other),copy()
        count(),mean(),sum(),min(),max(),variance(),sampleVariance(),stdev(),sampleStdev() 
    stdev()
        Compute the standard deviation of this RDD's elements.
        >>> sc.parallelize([1, 2, 3]).stdev()
        0.816...
    variance()
        Compute the variance of this RDD's elements.
        >>> sc.parallelize([1, 2, 3]).variance()
        0.666...
    sum()
        Add up the elements in this RDD.
        >>> sc.parallelize([1.0, 2.0, 3.0]).sum()
        6.0
    sumApprox(timeout, confidence=0.95)
        >>> rdd = sc.parallelize(range(1000), 10)
        >>> r = sum(range(1000))
        >>> abs(rdd.sumApprox(1000) - r) / r < 0.05
        True
    filter(f)
        Return a new RDD containing only the elements that satisfy a predicate.
        >>> rdd = sc.parallelize([1, 2, 3, 4, 5])
        >>> rdd.filter(lambda x: x % 2 == 0).collect()
        [2, 4]
    map(f, preservesPartitioning=False)
        Return a new RDD of f(e)
        >>> rdd = sc.parallelize(["b", "a", "c"])
        >>> sorted(rdd.map(lambda x: (x, 1)).collect())
        [('a', 1), ('b', 1), ('c', 1)]
    mapPartitions(f, preservesPartitioning=False)
        Return a new RDD by applying a function f(iterator)
        where iterator consists of all elements in that partition
        >>> rdd = sc.parallelize([1, 2, 3, 4], 2)
        >>> def f(iterator): yield sum(iterator)
        >>> rdd.mapPartitions(f).collect()
        [3, 7]
    mapPartitionsWithIndex(f, preservesPartitioning=False)
        Return a new RDD by applying a function f(index, Iterator)
        where iterator consists of all elements in that partition
        >>> rdd = sc.parallelize([1, 2, 3, 4], 4)
        >>> def f(splitIndex, iterator): yield splitIndex
        >>> rdd.mapPartitionsWithIndex(f).sum()
        6
    mapPartitionsWithSplit(f, preservesPartitioning=False)
        Deprecated: use mapPartitionsWithIndex instead.
    mapValues(f)
        For RDD of (K,V)
        Same as map, but for each key, transform values by f(v)
        >>> x = sc.parallelize([("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])])
        >>> def f(x): return len(x)
        >>> x.mapValues(f).collect()
        [('a', 3), ('b', 1)]
    flatMap(f, preservesPartitioning=False)
        Return a new RDD with f(e):Iterable and after flattening 
        >>> rdd = sc.parallelize([2, 3, 4])
        >>> sorted(rdd.flatMap(lambda x: range(1, x)).collect())
        [1, 1, 1, 2, 2, 3]
        >>> sorted(rdd.flatMap(lambda x: [(x, x), (x, x)]).collect())
        [(2, 2), (2, 2), (3, 3), (3, 3), (4, 4), (4, 4)]
    flatMapValues(f)
        For RDD of (K,V), mapping of Values 
        for each key, transform values by f(v):Iterable and flatening as (K,each_from_Iterable)
        >>> x = sc.parallelize([("a", ["x", "y", "z"]), ("b", ["p", "r"])])
        >>> def f(x): return x
        >>> x.flatMapValues(f).collect()
        [('a', 'x'), ('a', 'y'), ('a', 'z'), ('b', 'p'), ('b', 'r')]
    fold(zeroValue, op)
        op(r,e):type_of_r is applied with zeroValue , 
        note op(r,e) must be commutative, else result is not predictable
        >>> from operator import add
        >>> sc.parallelize([1, 2, 3, 4, 5]).fold(0, add)
        15
    foldByKey(zeroValue, func, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        For RDD of (K,V), same as above, but for same key , func(r,v):type_of_r
        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> from operator import add
        >>> sorted(rdd.foldByKey(0, add).collect())
        [('a', 2), ('b', 1)]
    reduce(f)
        Reduces the elements of this RDD using f(r,e), 
        must be commutative and associative
        >>> from operator import add
        >>> sc.parallelize([1, 2, 3, 4, 5]).reduce(add)
        15
        >>> sc.parallelize((2 for _ in range(10))).map(lambda x: 1).cache().reduce(add)
        10
    reduceByKey(func, numPartitions=None, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        For RDD of (K,V), same as above, but for same key, func(r,v)
        >>> from operator import add
        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.reduceByKey(add).collect())
        [('a', 2), ('b', 1)]
    reduceByKeyLocally(func)
        For RDD of (K,V), same as above, but for same key, func(r,v)
        but return the results immediately to the master as a dictionary.
        >>> from operator import add
        >>> rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
        >>> sorted(rdd.reduceByKeyLocally(add).items())
        [('a', 2), ('b', 1)]
    treeReduce(f, depth=2)
        Reduces the elements of this RDD in a multi-level tree pattern.
        depth – suggested depth of the tree (default: 2) 
        >>> add = lambda x, y: x + y
        >>> rdd = sc.parallelize([-5, -4, -3, -2, -1, 1, 2, 3, 4], 10)
        >>> rdd.treeReduce(add)
        -5
        >>> rdd.treeReduce(add, 1)
        -5
        >>> rdd.treeReduce(add, 2)
        -5
        >>> rdd.treeReduce(add, 5)
        -5
        >>> rdd.treeReduce(add, 10)
        -5
    sortBy(keyfunc, ascending=True, numPartitions=None)
        Sorts this RDD by the given keyfunc(e)
        >>> tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
        >>> sc.parallelize(tmp).sortBy(lambda x: x[0]).collect()
        [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
        >>> sc.parallelize(tmp).sortBy(lambda x: x[1]).collect()
        [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
    sortByKey(ascending=True, numPartitions=None, keyfunc=lambda x: x)
        For RDD of (K,V), sorts by keyfunc taking K and returning key for sorting 
        >>> tmp = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
        >>> sc.parallelize(tmp).sortByKey().first()
        ('1', 3)
        >>> sc.parallelize(tmp).sortByKey(True, 1).collect()
        [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
        >>> sc.parallelize(tmp).sortByKey(True, 2).collect()
        [('1', 3), ('2', 5), ('a', 1), ('b', 2), ('d', 4)]
        >>> tmp2 = [('Mary', 1), ('had', 2), ('a', 3), ('little', 4), ('lamb', 5)]
        >>> tmp2.extend([('whose', 6), ('fleece', 7), ('was', 8), ('white', 9)])
        >>> sc.parallelize(tmp2).sortByKey(True, 3, keyfunc=lambda k: k.lower()).collect()
        [('a', 3), ('fleece', 7), ('had', 2), ('lamb', 5),...('white', 9), ('whose', 6)]
    foreach(f)
        Applies a function to all elements of this RDD.
        >>> def f(x): print(x)
        >>> sc.parallelize([1, 2, 3, 4, 5]).foreach(f)
    foreachPartition(f)
        Applies a function, f(iterator) to each partition of this RDD.
        where iterator consists of all elements in that partition
        >>> def f(iterator):
                for x in iterator:
                    print(x)
        >>> sc.parallelize([1, 2, 3, 4, 5]).foreachPartition(f)    
    getNumPartitions()
        Returns the number of partitions in RDD
        >>> rdd = sc.parallelize([1, 2, 3, 4], 2)
        >>> rdd.getNumPartitions()
        2
    histogram(buckets)
        Compute a histogram using the provided buckets. 
        buckets [1,10,20,50] means the buckets are [1,10) [10,20) [20,50], 
        buckets can also be number of buckets
        The return value is a tuple of buckets and histogram(count of elements in that bucket).
        >>> rdd = sc.parallelize(range(51))
        >>> rdd.histogram(2)
        ([0, 25, 50], [25, 26])
        >>> rdd.histogram([0, 5, 25, 50])
        ([0, 5, 25, 50], [5, 20, 26])
        >>> rdd.histogram([0, 15, 30, 45, 60])  # evenly spaced buckets
        ([0, 15, 30, 45, 60], [15, 15, 15, 6])
        >>> rdd = sc.parallelize(["ab", "ac", "b", "bd", "ef"])
        >>> rdd.histogram(("a", "b", "c"))
        (('a', 'b', 'c'), [2, 2])
    isEmpty()
        Returns true if and only if the RDD contains no elements at all.
        >>> sc.parallelize([]).isEmpty()
        True
        >>> sc.parallelize([1]).isEmpty()
        False
    name()
        Return the name of this RDD.
    setName(name)
        Assign a name to this RDD.
        >>> rdd1 = sc.parallelize([1, 2])
        >>> rdd1.setName('RDD1').name()
        u'RDD1'
    pipe(command, env=None, checkCode=False)
        Return an RDD created by piping elements to a forked external process.
        >>> sc.parallelize(['1', '2', '', '3']).pipe('cat').collect()
        [u'1', u'2', u'', u'3']
    randomSplit(weights, seed=None)
        Randomly splits this RDD with the provided weights.
        >>> rdd = sc.parallelize(range(500), 1)
        >>> rdd1, rdd2 = rdd.randomSplit([2, 3], 17)
        >>> len(rdd1.collect() + rdd2.collect())
        500
        >>> 150 < rdd1.count() < 250
        True
        >>> 250 < rdd2.count() < 350
        True
    sample(withReplacement, fraction, seed=None)
        Return a sampled subset of this RDD.
        >>> rdd = sc.parallelize(range(100), 4)
        >>> 6 <= rdd.sample(False, 0.1, 81).count() <= 14
        True
    sampleByKey(withReplacement, fractions, seed=None)
        For RDD of (K,V)
        Return a subset of this RDD sampled by key
        Create a sample of this RDD using variable sampling rates for different keys 
        as specified by fractions, a key to sampling rate map.        
        >>> fractions = {"a": 0.2, "b": 0.1} #for key a, 20%, for key b, 10% 
        >>> rdd = sc.parallelize(fractions.keys()).cartesian(sc.parallelize(range(0, 1000)))
        >>> sample = dict(rdd.sampleByKey(False, fractions, 2).groupByKey().collect())
        >>> 100 < len(sample["a"]) < 300 and 50 < len(sample["b"]) < 150
        True
        >>> max(sample["a"]) <= 999 and min(sample["a"]) >= 0
        True
        >>> max(sample["b"]) <= 999 and min(sample["b"]) >= 0
        True
    takeSample(withReplacement, num, seed=None)
        Return a fixed-size sampled subset of this RDD.
        >>> rdd = sc.parallelize(range(0, 10))
        >>> len(rdd.takeSample(True, 20, 1))
        20
        >>> len(rdd.takeSample(False, 5, 2))
        5
        >>> len(rdd.takeSample(False, 15, 3))
        10
    saveAsHadoopDataset(conf, keyConverter=None, valueConverter=None)
        Output a Python RDD of(K, V) to any Hadoop file system, 
        using the old Hadoop OutputFormat API (mapred package). 
        •conf – Hadoop job configuration, passed in as a dict
        Keys/values are converted for output using either user specified converters 
        or, by default, org.apache.spark.api.python.JavaToWritableConverter
    saveAsNewAPIHadoopDataset(conf, keyConverter=None, valueConverter=None)
        Output a Python RDD of (K, V) to any Hadoop file system, 
        using the new Hadoop OutputFormat API (mapreduce package). 
        Keys/values are converted for output using either user specified converters 
        or, by default, org.apache.spark.api.python.JavaToWritableConverter
    saveAsHadoopFile(path, outputFormatClass, keyClass=None, valueClass=None, keyConverter=None, valueConverter=None, conf=None, compressionCodecClass=None)
        Output a Python RDD of (K, V) to any Hadoop file system, 
        using the old Hadoop OutputFormat API (mapred package)
        •path – path to Hadoop file
        •outputFormatClass – fully qualified classname of Hadoop OutputFormat (e.g. 'org.apache.hadoop.mapred.SequenceFileOutputFormat')
        •keyClass – fully qualified classname of key Writable class (e.g. 'org.apache.hadoop.io.IntWritable', None by default)
        •valueClass – fully qualified classname of value Writable class (e.g. 'org.apache.hadoop.io.Text', None by default)
        •keyConverter – (None by default)
        •valueConverter – (None by default)
        •conf – (None by default)
        •compressionCodecClass – (None by default)
    saveAsNewAPIHadoopFile(path, outputFormatClass, keyClass=None, valueClass=None, keyConverter=None, valueConverter=None, conf=None)
        Output a Python RDD of (K, V) to any Hadoop file system, 
        using the new Hadoop OutputFormat API (mapreduce package)
    saveAsPickleFile(path, batchSize=10)
        Save this RDD as a SequenceFile of serialized objects. 
        >>> tmpFile = tempfile.NamedTemporaryFile(delete=True)
        >>> tmpFile.close()
        >>> sc.parallelize([1, 2, 'spark', 'rdd']).saveAsPickleFile(tmpFile.name, 3)
        >>> sorted(sc.pickleFile(tmpFile.name, 5).map(str).collect())
        ['1', '2', 'rdd', 'spark']
    saveAsSequenceFile(path, compressionCodecClass=None)
        Output a Python RDD of (K, V)
        to any Hadoop file system including local filesystem,  
        using the org.apache.hadoop.io.Writable types that we convert from the RDD's key and value types
    saveAsTextFile(path, compressionCodecClass=None)
        Save this RDD as a text file, using string representations of elements.
        •path – path to text file
        •compressionCodecClass – (None by default) string i.e. 'org.apache.hadoop.io.compress.GzipCodec'
        >>> tempFile = tempfile.NamedTemporaryFile(delete=True)
        >>> tempFile.close()
        >>> sc.parallelize(range(10)).saveAsTextFile(tempFile.name)
        >>> from fileinput import input
        >>> from glob import glob
        >>> ''.join(sorted(input(glob(tempFile.name + "/part-0000*"))))
        '0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n'
    intersection(other)
        Return the intersection of this RDD and another one. 
        The output will not contain any duplicate elements, even if the input RDDs did.
        >>> rdd1 = sc.parallelize([1, 10, 2, 3, 4, 5])
        >>> rdd2 = sc.parallelize([1, 6, 2, 3, 7, 8])
        >>> rdd1.intersection(rdd2).collect()
        [1, 2, 3]
    subtract(other, numPartitions=None)
        Return each value in self that is not contained in other.
        >>> x = sc.parallelize([("a", 1), ("b", 4), ("b", 5), ("a", 3)])
        >>> y = sc.parallelize([("a", 3), ("c", None)])
        >>> sorted(x.subtract(y).collect())
        [('a', 1), ('b', 4), ('b', 5)]
    subtractByKey(other, numPartitions=None)
        For RDD of (K,V)
        Return each (key, value) pair in self that has no pair with matching key in other.
        >>> x = sc.parallelize([("a", 1), ("b", 4), ("b", 5), ("a", 2)])
        >>> y = sc.parallelize([("a", 3), ("c", None)])
        >>> sorted(x.subtractByKey(y).collect())
        [('b', 4), ('b', 5)]
    union(other)
        Return the union of this RDD and another one.
        >>> rdd = sc.parallelize([1, 1, 2, 3])
        >>> rdd.union(rdd).collect()
        [1, 1, 2, 3, 1, 1, 2, 3]
    toDebugString()
        A description of this RDD and its recursive dependencies for debugging.
    toLocalIterator()
        Return an iterator that contains all of the elements in this RDD.
        >>> rdd = sc.parallelize(range(10))
        >>> [x for x in rdd.toLocalIterator()]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    values()
        For RDD of (K,V)
        Return an RDD with the values 
        >>> m = sc.parallelize([(1, 2), (3, 4)]).values()
        >>> m.collect()
        [2, 4]
    keys()
        For RDD of (K,V)
        Return an RDD with the keys of each tuple.
        >>> m = sc.parallelize([(1, 2), (3, 4)]).keys()
        >>> m.collect()
        [1, 3]
    lookup(key)
        For RDD of (K,V)
        Return the list of values in the RDD for key key.
        >>> l = range(1000)
        >>> rdd = sc.parallelize(zip(l, l), 10)
        >>> rdd.lookup(42)  # slow
        [42]
        >>> sorted = rdd.sortByKey()
        >>> sorted.lookup(42)  # fast
        [42]
        >>> sorted.lookup(1024)
        []
        >>> rdd2 = sc.parallelize([(('a', 'b'), 'c')]).groupByKey()
        >>> list(rdd2.lookup(('a', 'b'))[0])
        ['c']
    zip(other)
        Zips this RDD with another one, Returns RDD of (E1,E2)
        >>> x = sc.parallelize(range(0,5))
        >>> y = sc.parallelize(range(1000, 1005))
        >>> x.zip(y).collect()
        [(0, 1000), (1, 1001), (2, 1002), (3, 1003), (4, 1004)]
    zipWithIndex()
        Zips this RDD with its element indices ie (e,index)
        first item in the first partition gets index 0, 
        and the last item in the last partition receives the largest index.
        >>> sc.parallelize(["a", "b", "c", "d"], 3).zipWithIndex().collect()
        [('a', 0), ('b', 1), ('c', 2), ('d', 3)]
    zipWithUniqueId()
        Zips this RDD with generated unique Long ids ie (e, id)
        Items in the kth partition will get ids k, n+k, 2*n+k, ..., 
        where n is the number of partitions. 
        >>> sc.parallelize(["a", "b", "c", "d", "e"], 3).zipWithUniqueId().collect()
        [('a', 0), ('b', 1), ('c', 4), ('d', 2), ('e', 5)]
    isCheckpointed()
        Return whether this RDD is checkpointed
    localCheckpoint()
        Mark this RDD for local checkpointing using Spark's existing caching layer.
    getCheckpointFile()
        Gets the name of the file to which this RDD was checkpointed
        Not defined if RDD is checkpointed locally.
    checkpoint()
        Mark this RDD for checkpointing via SparkContext.setCheckpointDir() 


##Spark - RDD - Printing elements of an RDD

#in local mode, OK 
#in cluster mode - stdout is executer/another machines stdout, hence NOT_OK 
rdd.foreach(println) 
rdd.map(println)

#solution -  use the collect() method to first bring the RDD to the driver node thus: 
rdd.collect().foreach(println). 

#This can cause the driver to run out of memory- hence safer is to use take 
rdd.take(100).foreach(println).


##Spark - RDD - Transformations

map(func)                   Return a new distributed dataset formed by passing each element of the source through a function func.  

filter(func)                Return a new dataset formed by selecting those elements of the source on which func returns true.  

flatMap(func)               Similar to map, but each input item can be mapped to 0 or more output items 
                            (so func should return a Seq rather than a single item).  

mapPartitions(func)         Similar to map, but runs separately on each partition (block) of the RDD, 
                            so func must be of type Iterator<T> => Iterator<U> when running on an RDD of type T.  
                            
mapPartitionsWithIndex(func)    Similar to mapPartitions, but also provides func with an integer value representing the index of the partition, 
                                so func must be of type (Int, Iterator<T>) => Iterator<U> when running on an RDD of type T.  
sample(withReplacement, fraction, seed)     Sample a fraction fraction of the data, with or without replacement, using a given random number generator seed.  

union(otherDataset)         Return a new dataset that contains the union of the elements in the source dataset and the argument.  
intersection(otherDataset)  Return a new RDD that contains the intersection of elements in the source dataset and the argument.  
distinct([numTasks]))       Return a new dataset that contains the distinct elements of the source dataset. 

groupByKey([numTasks])      When called on a dataset of (K, V) pairs, returns a dataset of (K, Iterable<V>) pairs. 
                            Note: If you are grouping in order to perform an aggregation (such as a sum or average) over each key, 
                            using reduceByKey or aggregateByKey will yield much better performance. 
                            Note: By default, the level of parallelism in the output depends on the number of partitions of the parent RDD. 
                            You can pass an optional numTasks argument to set a different number of tasks.  

reduceByKey(func, [numTasks])       When called on a dataset of (K, V) pairs, 
                                    returns a dataset of (K, V) pairs where the values for each key are aggregated 
                                    using the given reduce function func, 
                                    which must be of type (V,V) => V. 
                                    Like in groupByKey, the number of reduce tasks is configurable 
                                    through an optional second argument.  

aggregateByKey(zeroValue)(seqOp, combOp, [numTasks])   When called on a dataset of (K, V) pairs, 
                                                       returns a dataset of (K, U) pairs where the values 
                                                       for each key are aggregated using the given combine functions 
                                                       and a neutral "zero" value. 
                                                       Allows an aggregated value type that is different than the input value type, 
                                                       while avoiding unnecessary allocations. 
                                                       Like in groupByKey, the number of reduce tasks is configurable 
                                                       through an optional second argument.  

sortByKey([ascending], [numTasks])  When called on a dataset of (K, V) pairs where K implements Ordered, 
                                    returns a dataset of (K, V) pairs sorted by keys in ascending or descending order, as specified in the boolean ascending argument. 

join(otherDataset, [numTasks])      When called on datasets of type (K, V) and (K, W), 
                                    returns a dataset of (K, (V, W)) pairs with all pairs of elements for each key. 
                                    Outer joins are supported through leftOuterJoin, rightOuterJoin, and fullOuterJoin.  
                                    
cogroup(otherDataset, [numTasks])   When called on datasets of type (K, V) and (K, W), 
                                    returns a dataset of (K, (Iterable<V>, Iterable<W>)) tuples. 
                                    This operation is also called groupWith.  
                                    
cartesian(otherDataset)             When called on datasets of types T and U, 
                                    returns a dataset of (T, U) pairs (all pairs of elements).  

pipe(command, [envVars])            Pipe each partition of the RDD through a shell command, 
                                    e.g. a Perl or bash script. 
                                    RDD elements are written to the process stdin 
                                    and lines output to its stdout are returned as an RDD of strings.  
                                    
coalesce(numPartitions)             Decrease the number of partitions in the RDD to numPartitions. 
                                    Useful for running operations more efficiently after filtering down a large dataset.  
                                    
repartition(numPartitions)          Reshuffle the data in the RDD randomly to create either more or fewer partitions and balance it across them. 
                                    This always shuffles all data over the network.  
                                    
repartitionAndSortWithinPartitions(partitioner)  Repartition the RDD according to the given partitioner 
                                                 and, within each resulting partition, sort records by their keys. This is more efficient than calling repartition and then sorting within each partition because it can push the sorting down into the shuffle machinery.  



##Spark - RDD - Actions

reduce(func)    Aggregate the elements of the dataset using a function func (which takes two arguments and returns one). 
                The function should be commutative and associative so that it can be computed correctly in parallel.  

collect()       Return all the elements of the dataset as an array at the driver program. This is usually useful after a filter or other operation that returns a sufficiently small subset of the data.  
count()         Return the number of elements in the dataset.  
first()         Return the first element of the dataset (similar to take(1)).  
take(n)         Return an array with the first n elements of the dataset.  

takeSample(withReplacement, num, [seed])  Return an array with a random sample of num elements of the dataset, 
                                          with or without replacement, optionally pre-specifying a random number generator seed. 

                                          takeOrdered(n, [ordering])  Return the first n elements of the RDD using either their natural order or a custom comparator.  

saveAsTextFile(path)        Write the elements of the dataset as a text file (or set of text files) in a given directory in the local filesystem, HDFS or any other Hadoop-supported file system. Spark will call toString on each element to convert it to a line of text in the file.  
saveAsSequenceFile(path)    Write the elements of the dataset as a Hadoop SequenceFile in a given path in the local filesystem, HDFS or any other Hadoop-supported file system. This is available on RDDs of key-value pairs that implement Hadoop Writable interface. In Scala, it is also available on types that are implicitly convertible to Writable (Spark includes conversions for basic types like Int, Double, String, etc).  
saveAsObjectFile(path)      Write the elements of the dataset in a simple format using Java serialization, which can then be loaded using SparkContext.objectFile().  

countByKey()                Only available on RDDs of type (K, V). 
                            Returns a hashmap of (K, Int) pairs with the count of each key.  

foreach(func)               Run a function func on each element of the dataset. 
                            This is usually done for side effects such as updating an Accumulator or interacting with external storage systems. 
                            Note: modifying variables other than Accumulators outside of the foreach() may result in undefined behavior. See Understanding closures  for more details. 










##Spark - RDD - Passing Functions in the driver program to run on the cluster. 
#IMP*** functions must be serialized/pickled - else ERROR 

#Fllowings are OK 
    •Lambda expressions, for simple functions that can be written as an expression. 
    •Local defs inside a function calling Spark methods
    •Top-level functions in a module.

#Example 
"""MyScript.py"""
if __name__ == "__main__":
    def myFunc(s):
        words = s.split(" ")
        return len(words)

    sc = SparkContext(...)
    sc.textFile("file.txt").map(myFunc)
    
#For below type of cases, whole object needs to be sent to cluster 
class MyClass(object):
    def func(self, s):
        return s
    def doStuff(self, rdd):
        return rdd.map(self.func)
#OR 
class MyClass(object):
    def __init__(self):
        self.field = "Hello"
    def doStuff(self, rdd):
        return rdd.map(lambda s: self.field + s)

#Solution - copy field into a local variable 
def doStuff(self, rdd):
    field = self.field
    return rdd.map(lambda s: field + s)



##Spark - RDD - Mutating variables 
#RDD operations that MODIFY variables outside of their scope - WRONG, 
#Use accumulators


#WRONG Usage - this might work in local mode, but fails in other 
counter = 0
rdd = sc.parallelize(data)

# Wrong: Donot do this!!
def increment_counter(x):
    global counter
    counter += x
rdd.foreach(increment_counter)

print("Counter value: ", counter)


##Spark - RDD - Shared Variables - Accumulators
class pyspark.Accumulator(aid, value, accum_param)
    A shared variable that can be accumulated, 
    i.e., has a commutative and associative 'add' operation. 
    Worker tasks on a Spark cluster can add values to an Accumulator with the += operator, 
    but only the driver program is allowed to access its value, using value. 
        add(term)
            Adds a term to this accumulator's value
        value
            Get the accumulator's value; only usable in driver program

 
#Spark natively supports accumulators of numeric types, 
#and programmers can add support for new types.


accum = sc.accumulator(0) #Driver
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x)) #Executor
accum.value  #10  #Driver 



##User defined accumulator 
#The AccumulatorParam interface has two methods: zero for providing a 'zero value' for your data type, 
#and addInPlace for adding two values together

#Example - handling vectors as accumulator 
from pyspark.util import AccumulatorParam
from pyspark.mllib.linalg import Vectors 
class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return Vectors.zeros(initialValue.size)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1

# Then, create an Accumulator of this type:
vecAccum = sc.accumulator(Vectors.dense([1.0, 2.0]), VectorAccumulatorParam())#Driver
sc.parallelize([1, 2, 3, 4]).foreach(lambda x: vecAccum.add([x,x])) #Executor
vecAccum.value    #Driver 




##Spark - RDD - Shared Variables - Broadcast Variables


class pyspark.Broadcast(sc=None, value=None, pickle_registry=None, path=None)
    Broadcast variables allow the programmer to keep a read-only variable cached 
    on each machine rather than shipping a copy of it with tasks. 
    Once created by Driver, 
    it can  be used(.value) in any functions that run on the cluster 
    destroy()
        Destroy all data and metadata related to this broadcast variable. 
    dump(value, f)
    load(path)
    unpersist(blocking=False)
    value
        Return the broadcasted value
    
    

            
#Example 
>>> from pyspark.context import SparkContext
>>> sc = SparkContext('local', 'test')
>>> b = sc.broadcast([1, 2, 3, 4, 5])
>>> b.value
[1, 2, 3, 4, 5]
>>> sc.parallelize([0, 0]).flatMap(lambda x: b.value).collect()
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
>>> b.unpersist()
>>> large_broadcast = sc.broadcast(range(10000))





##Spark - RDD - Shuffle operations
#The shuffle is Spark's mechanism for re-distributing data so that it's grouped differently across partitions. 
#This typically involves copying data across executors and machines, making the shuffle a complex and costly operation.

#Operations which can cause a shuffle include repartition operations 
#repartition and coalesce, 'ByKey operations (except for counting) 
#groupByKey and reduceByKey, and join operations like cogroup and join.



#Although the set of elements in each partition of newly shuffled data will be deterministic, 
#and so is the ordering of partitions themselves, 
#the ordering of these elements is not. 
#If one desires predictably ordered data following shuffle , use below 
    •mapPartitions to sort each partition using, for example, 'sorted'
    •repartitionAndSortWithinPartitions to efficiently sort partitions while simultaneously repartitioning
    •sortBy to make a globally ordered RDD

#Certain shuffle operations can consume significant amounts of heap memory 
#since they employ in-memory data structures to organize records before or after transferring them
#Shuffle also generates a large number of intermediate files on disk




##Spark - RDD - Persistence

#You can mark an RDD to be persisted using the persist() or cache() 
#The first time it is computed in an action, it will be kept in memory on the nodes.

#The cache() method is a shorthand for using the default storage level, which is StorageLevel.MEMORY_ONLY 

#Spark automatically monitors cache usage on each node 
#and drops out old data partitions in a least-recently-used (LRU) fashion. 
#OR use the RDD.unpersist() method.

#The full set of storage levels is:
MEMORY_ONLY         Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory, some partitions will not be cached and will be recomputed on the fly each time they are needed. This is the default level.  
MEMORY_AND_DISK     Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory, store the partitions that don't fit on disk, and read them from there when they're needed.  
MEMORY_ONLY_SER     (Java and Scala)  Store RDD as serialized Java objects (one byte array per partition). This is generally more space-efficient than deserialized objects, especially when using a fast serializer, but more CPU-intensive to read.  
MEMORY_AND_DISK_SER (Java and Scala)  Similar to MEMORY_ONLY_SER, but spill partitions that don't fit in memory to disk instead of recomputing them on the fly each time they're needed.  
DISK_ONLY           Store the RDD partitions only on disk.  
MEMORY_ONLY_2, MEMORY_AND_DISK_2, etc.  Same as the levels above, but replicate each partition on two cluster nodes.  
OFF_HEAP (experimental)     Similar to MEMORY_ONLY_SER, but store the data in off-heap memory. This requires off-heap memory to be enabled.  


##Which Storage Level to Choose

•If your RDDs fit comfortably with the default storage level (MEMORY_ONLY), leave them that way. 
 This is the most CPU-efficient option, allowing operations on the RDDs to run as fast as possible.

•If not, try using MEMORY_ONLY_SER and selecting a fast serialization library to make the objects much more space-efficient, 
 but still reasonably fast to access. (Java and Scala)

•Don't spill to disk unless the functions that computed your datasets are expensive, 
 or they filter a large amount of the data. 
 Otherwise, recomputing a partition may be as fast as reading it from disk.

•Use the replicated storage levels if you want fast fault recovery 
 (e.g. if using Spark to serve requests from a web application). 
 All the storage levels provide full fault tolerance by recomputing lost data, 
 but the replicated ones let you continue running tasks on the RDD without waiting to recompute a lost partition.














            

        
            
            

/***** DataFrame (ML) ****/
#sql-programming-guide.md - http://spark.apache.org/docs/latest/sql-programming-guide.html


###Spark - DataFrames

#A DataFrame is a dataset organized into named columns(available in python, like pandas.DataFrame)
#ie a 2D struture of ROW x Column, Columns are called features and rows are observations

###Spark - DataFrames - pyspark.sql module 

•pyspark.sql.SparkSession   Main entry point for DataFrame and SQL functionality.
•pyspark.sql.DataFrame      A distributed collection of data grouped into named columns.
•pyspark.sql.Column         A column expression in a DataFrame.
•pyspark.sql.Row            A row of data in a DataFrame.
•pyspark.sql.GroupedData    Aggregation methods, returned by DataFrame.groupBy().

•pyspark.sql.DataFrameNaFunctions       Methods for handling missing data (null values).
•pyspark.sql.DataFrameStatFunctions     Methods for statistics functionality.
•pyspark.sql.functions                  List of built-in functions available for DataFrame.
•pyspark.sql.types                      List of data types available.
•pyspark.sql.Window                     For working with window functions.


###Spark - DataFrames - SparkSession 

#As of Spark 2.0, SQLContext is replaced by SparkSession
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
#Example 
val warehouseLocation = "file:${system:user.dir}/spark-warehouse" 


val spark = SparkSession 
 .builder() 
 .appName("SparkSessionZipsExample") 
 .config("spark.sql.warehouse.dir", warehouseLocation) 
 .enableHiveSupport() 
 .getOrCreate() 

    

###Spark - DataFrames - SparkSession - Reference 
rdd= rdd = sc.parallelize(
        [Row(field1=1, field2="row1"),
         Row(field1=2, field2="row2"),
         Row(field1=3, field2="row3")])
df = rdd.toDF()

    
class Builder
    Builder for SparkSession. Get it from SparkSession.builder
    appName(name)
        Sets a name for the application
    config(key=None, value=None, conf=None)
        Sets a config option.
        >>> from pyspark.conf import SparkConf
        >>> SparkSession.builder.config(conf=SparkConf())        
        >>> SparkSession.builder.config("spark.some.config.option", "some-value")
        <pyspark.sql.session...
    enableHiveSupport()
        Enables Hive support, including connectivity to a persistent Hive metastore, 
        support for Hive serdes, and Hive user-defined functions.
    getOrCreate()
        Gets an existing SparkSession or, if there is no existing one, 
        creates a new one based on the options set in this builder.
        >>> s1 = SparkSession.builder.config("k1", "v1").getOrCreate()
        >>> s1.conf.get("k1") == s1.sparkContext.getConf().get("k1") == "v1"
        True
    master(master)
        Sets the Spark master URL to connect to, such as 'local' to run locally, 
        'local[4]' to run locally with 4 cores, or 'spark://master:7077' to run on a Spark standalone cluster.



class pyspark.sql.SparkSession(sparkContext, jsparkSession=None)
    The entry point to programming Spark with  DataFrame API.
    A SparkSession can be used create DataFrame, 
    register DataFrame as tables, execute SQL over tables, 
    cache tables, and read many datasource (eg parquet) files. 
    SparkSession provides builtin support for Hive features
    Can be used with 'with' syntax 
        with SparkSession.builder.(...).getOrCreate() as session:
            ....            
    #Attributes 
    classAttribute builder = <pyspark.sql.session.Builder object at 0x7fc358d6e250>
    catalog
        Interface through which the user may create, drop, alter 
        or query underlying databases, tables, functions etc.
        #fetch metadata data from the catalog
        spark.catalog.listDatabases.show(false)
        spark.catalog.listTables.show(false)
    conf
        Instance of SparkConf, Runtime configuration interface for Spark
        #set new runtime options
        spark.conf.set("spark.sql.shuffle.partitions", 6)
        spark.conf.set("spark.executor.memory", "2g")
        #get all settings
        configMap = spark.conf.getAll() #dict[String, String]
    newSession()
        Returns a new SparkSession as new session, 
        that has separate SQLConf, registered temporary views and UDFs, 
        but shared SparkContext and table cache.
    createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)
        Creates a DataFrame from an RDD, a list or a pandas.DataFrame.
        >>> import pandas as pd
        >>> pd_df = pd.DataFrame({'a':[1,2,3,4], "b":[5,6,7,8]})        
        >>> pd_df
           a  b
        0  1  5
        1  2  6
        2  3  7
        3  4  8
        >>> spark.createDataFrame(pd_df)
        DataFrame[a: bigint, b: bigint]
        >>> spark.createDataFrame(pd_df).show()
        +---+---+
        |  a|  b|
        +---+---+
        |  1|  5|
        |  2|  6|
        |  3|  7|
        |  4|  8|
        +---+---+
        # create a DataFrame using spark.createDataFrame from a List or Seq
        langPercentDF = spark.createDataFrame([("Scala", 35), ("Python", 30), ("R", 15), ("Java", 20)])
        >>> langPercentDF
        DataFrame[_1: string, _2: bigint]
        #rename the columns
        lpDF = langPercentDF.withColumnRenamed("_1", "language").withColumnRenamed("_2", "percent")
        #order the DataFrame in descending order of percentage
        lpDF.orderBy(desc("percent")).show(false)
    range(start, end=None, step=1, numPartitions=None)
        Create a DataFrame with single pyspark.sql.types.LongType column named 'id',
        containing elements in a range from start to end (exclusive) with step value step.
        >>> spark.range(1, 7, 2).collect()
        [Row(id=1), Row(id=3), Row(id=5)]
        >>> spark.range(3).collect()
        [Row(id=0), Row(id=1), Row(id=2)]
        #create a Dataset using spark.range starting from 5 to 100, with increments of 5
        numDS = spark.range(5, 100, 5)
        # reverse the order and display first 5 items
        from pyspark.sql.functions  import * 
        numDS.orderBy(desc("id")).show(5)
        #compute descriptive stats and display them
        numDs.describe().show()        
    read
        Returns a DataFrameReader that can be used to read data in as a DataFrame.
        #example 
        zipsDF = spark.read.json("zips.json")
        #filter all cities whose population > 40K
        zipsDF.filter(zipsDF.pop > 40000).show(10)
        zipsDF.createOrReplaceTempView("zips_table")
        zipsDF.cache()
        resultsDF = spark.sql("SELECT city, pop, state, zip FROM zips_table")
        resultsDF.show(10)
        #drop the table if exists to get around existing table error
        spark.sql("DROP TABLE IF EXISTS zips_hive_table")
        #Read one table and save as a different hive table, hive should be enabled 
        spark.table("zips_table").write.saveAsTable("zips_hive_table")
        #make a similar query against the hive table 
        val resultsHiveDF = spark.sql("SELECT city, pop, state, zip FROM zips_hive_table WHERE pop > 40000")
        resultsHiveDF.show(10)
    readStream
        Returns a DataStreamReader that can be used to read data streams as a streaming DataFrame.
    sparkContext
        Returns the underlying SparkContext.
    sql(sqlQuery)
        Returns a DataFrame representing the result of the given query.
        >>> df.createOrReplaceTempView("table1")
        >>> df2 = spark.sql("SELECT field1 AS f1, field2 as f2 from table1")
        >>> df2.collect()
        [Row(f1=1, f2=u'row1'), Row(f1=2, f2=u'row2'), Row(f1=3, f2=u'row3')]
    stop()
        Stop the underlying SparkContext.
    streams
        Returns a StreamingQueryManager that allows managing all the StreamingQuery StreamingQueries active on this context.
    table(tableName)
        Returns the specified table as a DataFrame.
        >>> df.createOrReplaceTempView("table1")
        >>> df2 = spark.table("table1")
        >>> sorted(df.collect()) == sorted(df2.collect())
        True
    udf
        Returns a instance UDFRegistration for UDF registration.
        Use pyspark.sql.functions.udf for simplicity 
    version
        The version of Spark on which this application is running.
        
        

class pyspark.sql.SQLContext(sparkContext, sparkSession=None, jsqlContext=None)
    Deprecated, Use SparkSession.
        #set up the spark configuration and create contexts
        val sparkConf = SparkConf().setAppName("SparkSessionZipsExample").setMaster("local")
        #your handle to SparkContext to access other context like SQLContext
        sc = new SparkContext(sparkConf).set("spark.some.config.option", "some-value")
        sqlContext = pyspark.sql.SQLContext(sc)
    cacheTable(tableName)
        Caches the specified table in-memory.
    clearCache()
        Removes all cached tables from the in-memory cache.
    createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)
        Creates a DataFrame from an RDD, a list or a pandas.DataFrame.
    createExternalTable(tableName, path=None, source=None, schema=None, **options)
        Creates an external table based on the dataset in a data source.
        It returns the DataFrame associated with the external table.
    dropTempTable(tableName)
        Remove the temp table from catalog.
    getConf(key, defaultValue=None)
        Returns the value of Spark SQL configuration property for the given key.
        >>> sqlContext.getConf("spark.sql.shuffle.partitions")
        u'200'
        >>> sqlContext.getConf("spark.sql.shuffle.partitions", u"10")
        u'10'
        >>> sqlContext.setConf("spark.sql.shuffle.partitions", u"50")
        >>> sqlContext.getConf("spark.sql.shuffle.partitions", u"10")
        u'50'
    classmethod getOrCreate(sc)
        Get the existing SQLContext or create a new one with given SparkContext.
    newSession()
        Returns a new SQLContext as new session,
    range(start, end=None, step=1, numPartitions=None)
        Create a DataFrame with single pyspark.sql.types.LongType column named id, 
        containing elements in a range from start to end (exclusive) with step value step.
    read
        Returns a DataFrameReader that can be used to read data in as a DataFrame.
    readStream
        Returns a DataStreamReader that can be used to read data streams as a streaming DataFrame.
    registerDataFrameAsTable(df, tableName)
        Registers the given DataFrame as a temporary table in the catalog.
    registerFunction(name, f, returnType=StringType)
        Registers a python function (including lambda function) as a UDF 
    registerJavaFunction(name, javaClassName, returnType=None)
        Register a java UDF so it can be used in SQL statements.
    setConf(key, value)
        Sets the given Spark SQL configuration property.
    sql(sqlQuery)
        Returns a DataFrame representing the result of the given query.
    streams
        Returns a StreamingQueryManager that allows managing all the StreamingQuery StreamingQueries active on this context.
    table(tableName)
        Returns the specified table as a DataFrame.
    tableNames(dbName=None)
        Returns a list of names of tables in the database dbName.
    tables(dbName=None)
        Returns a DataFrame containing names of tables in the given database.
    udf
        Returns a UDFRegistration for UDF registration.
    uncacheTable(tableName)
        Removes the specified table from the in-memory cache.

    
class pyspark.sql.HiveContext(sparkContext, jhiveContext=None)
    Deprecated in 2.0.0. 
    Use SparkSession.builder.enableHiveSupport().getOrCreate().
        refreshTable(tableName)
            Invalidate and refresh all the cached the metadata of the given table. 
  
    
    
    
    
    
    
##Spark - DataFrames - NaN Semantics
#There is specially handling for not-a-number (NaN) when dealing with float or double types 
#that does not exactly match standard floating point semantics. 
•NaN = NaN returns true.
•In aggregations all NaN values are grouped together.
•NaN is treated as a normal value in join keys.
•NaN values go last when in ascending order, larger than any other numeric value.


##Spark - DataFrames - Using Row 
#A row in DataFrame 
#The fields in it can be accessed:
#•like attributes (row.key)
#•like dictionary values (row[key])

#For missing named argument,  explicitly set to None 

from pyspark.sql import * 
>>> Row.__mro__
(<class 'pyspark.sql.types.Row'>, <class 'tuple'>, <class 'object'>)

>>> row = Row(name="Alice", age=11)
>>> row
Row(age=11, name='Alice')
>>> row['name'], row['age']
('Alice', 11)
>>> row.name, row.age
('Alice', 11)
>>> 'name' in row
True
>>> 'wrong_key' in row
False

#Row also can be used to create another Row like class, 
#then it could be used to create Row objects
>>> Person = Row("name", "age")
>>> Person
<Row(name, age)>
>>> 'name' in Person
True
>>> 'wrong_key' in Person
False
>>> Person("Alice", 11)
Row(name='Alice', age=11)

#Methods 
asDict(recursive=False)
    Return as an dict
    Parameters:
        recursive – turns the nested Row as dict (default: False). 


>>> Row(name="Alice", age=11).asDict() == {'name': 'Alice', 'age': 11}
True
>>> row = Row(key=1, value=Row(name='a', age=2))
>>> row.asDict() == {'key': 1, 'value': Row(age=2, name='a')}
True
>>> row.asDict(True) == {'key': 1, 'value': {'name': 'a', 'age': 2}}
True


##Spark - DataFrame - Creation of DF - With a SparkSession, 
#applications can create DataFrames from an existing RDD, from a Hive table, 
#or from Spark data sources

SparkSession.createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)
    Creates a DataFrame 
    •data – an RDD of any kind of SQL data representation
            (e.g. row, tuple, int, boolean, etc.), 
            or list of tuples(as rows), or list of dict(as rows) or pandas.DataFrame.
    •schema – a pyspark.sql.types.DataType 
             or a datatype string 
             or a list of column names
             default is None. 
             The data type string format equals to pyspark.sql.types.DataType.simpleString, 
             except that top level struct type can omit the struct<> 
             and atomic types use typeName() as their format, 
             e.g. use byte instead of tinyint for pyspark.sql.types.ByteType. 
             We can also use int as a short name for IntegerType.
    •samplingRatio – the sample ratio of rows used for inferring
    •verifySchema – verify data types of every row against schema.
 
#Example  
#from list of tuples( as rows)
l = [('Alice', 1)]
spark.createDataFrame(l).collect()  #[Row(_1=u'Alice', _2=1)]
spark.createDataFrame(l, ['name', 'age']).collect()   ##[Row(name=u'Alice', age=1)]

#from list of dicts( as rows)
d = [{'name': 'Alice', 'age': 1}]
spark.createDataFrame(d).collect()  #[Row(age=1, name=u'Alice')]

#from RDD and list of column namees
rdd = sc.parallelize(l)
spark.createDataFrame(rdd).collect() #[Row(_1=u'Alice', _2=1)]
df = spark.createDataFrame(rdd, ['name', 'age']) 
df.collect() #[Row(name=u'Alice', age=1)]
df.printSchema()
df.show()

#With explicit schema 
from pyspark.sql.types import *
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)])
rdd = sc.parallelize(l)
df3 = spark.createDataFrame(rdd, schema)
df3.collect()  #[Row(name=u'Alice', age=1)]

#With schema as string 
rdd = sc.parallelize(l)
spark.createDataFrame(rdd, "a: string, b: int").collect()  #[Row(a=u'Alice', b=1)]

#From RDD of Row 
from pyspark.sql import Row
Person = Row('name', 'age')
rdd = sc.parallelize(l)
person = rdd.map(lambda r: Person(*r))
df2 = spark.createDataFrame(person)
df2.collect()  #[Row(name=u'Alice', age=1)]

#from pandas 
spark.createDataFrame(df.toPandas()).collect()    #[Row(name=u'Alice', age=1)]
spark.createDataFrame(pandas.DataFrame([[1, 2]])).collect()   #[Row(0=1, 1=2)]


# accepatble Python types for  Spark SQL DataType
_acceptable_types = {
    BooleanType: (bool,),
    ByteType: (int, long),
    ShortType: (int, long),
    IntegerType: (int, long),
    LongType: (int, long),
    FloatType: (float,),
    DoubleType: (float,),
    DecimalType: (decimal.Decimal,),
    StringType: (str, unicode),
    BinaryType: (bytearray,),
    DateType: (datetime.date, datetime.datetime),
    TimestampType: (datetime.datetime,),
    ArrayType: (list, tuple, array.array),
    MapType: (dict,),
    StructType: (tuple, list, dict),
}
# Mapping Python types to Spark SQL DataType
_type_mappings = {
    type(None): NullType,
    bool: BooleanType,
    int: LongType,
    float: DoubleType,
    str: StringType,
    bytearray: BinaryType,
    decimal.Decimal: DecimalType,
    datetime.date: DateType,
    datetime.datetime: TimestampType,
    datetime.time: TimestampType,
}
#Note python constructors 
decimal.Decimal([value[, context]])
    value can be an integer, string, tuple, float, or another Decimal object
    If value is a string, it should conform to the decimal numeric string syntax 
    after leading and trailing whitespace characters are removed:
    If value is a tuple, it should have three components, 
    a sign (0 for positive or 1 for negative), a tuple of digits, 
    and an integer exponent. 
    For example, Decimal((0, (1, 4, 1, 4), -3)) returns Decimal('1.414').
    If value is a float, the binary floating point value is losslessly converted to its exact decimal equivalent. 
    This conversion can often require 53 or more digits of precision. 
    For example, Decimal(float('1.1')) converts to Decimal('1.100000000000000088817841970012523233890533447265625').

datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
datetime.date(year, month, day)    
datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]])
datetime.time([hour[, minute[, second[, microsecond[, tzinfo]]]]])

#String of Datatype 
The data type string format equals DataType.simpleString
Atomic types can use DataType.typeName()(=lower case of class_name[:-4] ie byte for ByteType)
DataType.simpleString 
    by default = DataType.typeName(), 
    except below for atomic type 
    IntegerType     "integer" or 'int' 
    DecimalType     'decimal' or 'decimal(precision, scale)'
    ByteType        'byte' or 'tinyint'
    LongType        'long' or 'bigint'
Complex type can have following 
    StructType  "struct<field_name:field_type,...>"  or "field_name:field_type, .."
    MapType     "map<k_type_string, v_type_string>"
    ArrayType   "array<element_type>"

>>> pyspark.sql.types._all_atomic_types
{'float': <class 'pyspark.sql.types.FloatType'>, 
'timestamp': <class 'pyspark.sql.types.TimestampType'>, 
'double': <class 'pyspark.sql.types.DoubleType'>, 
'null': <class 'pyspark.sql.types.NullType'>, 
'short': <class 'pyspark.sql.types.ShortType'>, 
'binary': <class 'pyspark.sql.types.BinaryType'>, 
'byte': <class 'pyspark.sql.types.ByteType'>, 
'integer': <class 'pyspark.sql.types.IntegerType'>, 
'decimal': <class 'pyspark.sql.types.DecimalType'>, 
'boolean': <class 'pyspark.sql.types.BooleanType'>, 
'long': <class 'pyspark.sql.types.LongType'>, 
'string': <class 'pyspark.sql.types.StringType'>, 
'date': <class 'pyspark.sql.types.DateType'>}
>>> pyspark.sql.types._all_complex_types
{'struct': <class 'pyspark.sql.types.StructType'>, 
'map': <class 'pyspark.sql.types.MapType'>, 
'array': <class 'pyspark.sql.types.ArrayType'>}

>>> pyspark.sql.types._parse_datatype_string("int ")
IntegerType
>>> pyspark.sql.types._parse_datatype_string("a: byte, b: decimal(  16 , 8   ) ")
StructType([StructField(a,ByteType,true),StructField(b,DecimalType(16,8),true)])
>>> pyspark.sql.types._parse_datatype_string("a: array< short>")
StructType([StructField(a,ArrayType(ShortType,true),true)])
>>> pyspark.sql.types._parse_datatype_string(" map<string , string > ")
MapType(StringType,StringType,true)



#reference 
pyspark.sql.types module
    DataType
        Base class for all data types
            fromJson(json)
            jsonValue()
            json()
                Gets json string 
            needConversion()
            simpleString()
            toInternal(obj)
                Converts a Python object into an internal SQL object
            fromInternal(obj)
                Converts an internal SQL object into a native Python object.
            classmethod typeName()
            ==, !=
    NullType
    StringType
    BinaryType
    BooleanType
    DateType
    TimestampType
    DecimalType(precision=10, scale=0)
    DoubleType
    FloatType
    ByteType
    IntegerType
    LongType
    ShortType    
    ArrayType(elementType, containsNull=True)
        Array data type.
        fromInternal(obj)
        classmethod fromJson(json)
            from json string 
        jsonValue()
            to json object 
        josn()
            to json string 
        needConversion()
        simpleString()
        toInternal(obj)
    MapType(keyType, valueType, valueContainsNull=True)
        Map data type.
        fromInternal(obj)
        classmethod fromJson(json)
            from json string 
        jsonValue()
            to json object 
        josn()
            to json string 
        needConversion()
        simpleString()
        toInternal(obj)
    StructField(name, dataType, nullable=True, metadata=None)
        fromInternal(obj)
        classmethod fromJson(json)
            from json string 
        jsonValue()
            to json object 
        josn()
            to json string 
        needConversion()
        simpleString()
        toInternal(obj)
    StructType(fields=None) 
        List of StructField
        >>> struct1 = StructType([StructField("f1", StringType(), True)])
        >>> struct1["f1"]
        StructField(f1,StringType,true)
        >>> struct1[0]
        StructField(f1,StringType,true)
        #Methods 
        fromInternal(obj)
        classmethod fromJson(json)
            from json string 
        jsonValue()
            to json object 
        josn()
            to json string 
        needConversion()
        simpleString()
        toInternal(obj)        
        add(field, data_type=None, nullable=True, metadata=None)
            Construct a StructType by adding new elements to it to define the schema. 
            The method accepts either:
                a.A single parameter which is a StructField object.
                b.Between 2 and 4 parameters as (name, data_type, nullable (optional), metadata(optional). 
            The data_type parameter may be either a String or a DataType object.
            >>> struct1 = StructType().add("f1", StringType(), True).add("f2", StringType(), True, None)
            >>> struct2 = StructType([StructField("f1", StringType(), True), \
            ...     StructField("f2", StringType(), True, None)])
            >>> struct1 == struct2
            True
            >>> struct1 = StructType().add(StructField("f1", StringType(), True))
            >>> struct2 = StructType([StructField("f1", StringType(), True)])
            >>> struct1 == struct2
            True
            >>> struct1 = StructType().add("f1", "string", True)
            >>> struct2 = StructType([StructField("f1", StringType(), True)])
            >>> struct1 == struct2
            True

            
##Spark - DataFrame - DF to rdd 
df.rdd 

##Spark - DataFrame - Conversion from RDDs
#Spark SQL supports two different methods for converting existing RDDs into DataFrame
 
RDD.toDF(schema=None, sampleRatio=None)
SparkSession.createDataFrame(data, schema=None, samplingRatio=None, verifySchema=True)
    schema: a pyspark.sql.types.StructType eg StructType([StructField("name", StringType(), True),StructField("age", IntegerType(), True)])
            or list of names of Columns eg ['name', 'age']
            or string of datatype eg "name: string, age: int"


#METHOD-1 : Infer the schema
from pyspark.sql import Row
#row = Michael, 29
parts = spark.sparkContext.textFile(r"D:\Desktop\PPT\spark\data\people.txt").map(lambda l: l.split(","))
#[['Michael', ' 29'], ['Andy', ' 30'], ['Justin', ' 19']]
people = parts.map(lambda p: Row(name=p[0], age=int(p[1]))).toDF()
>>> people  #column name comes from Row 
DataFrame[age: bigint, name: string]


#METHOD-2: Programmatically Specifying the Schema

#another way , load the file 
sc = spark.sparkContext
lines = sc.textFile(r"D:\Desktop\PPT\spark\data\people.txt")
parts = lines.map(lambda l: l.split(","))
people = parts.map(lambda p: (p[0], p[1].strip()))

# The schema is encoded in a string.
schemaString = "name age"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)

# Apply the schema to the RDD.
peopleDF = spark.createDataFrame(people, schema)
#or 
peopleDF =  people.toDF(schema)  


##Spark - DataFrame - Userdefined function to operate on each element of a DF column
#Note f takes each element of a Column, not full Column together

class pyspark.sql.UDFRegistration(sqlContext)
    Wrapper for user-defined function registration.
    Get instance from sparkSession.udf 
    Use pyspark.sql.functions.udf for simplicity 
    #Attributes 
        register(name, f, returnType=StringType)
            Registers a python function (including lambda function) as a UDF
            
#Example  
>>> from pyspark.sql.types import IntegerType
>>> sqlContext.udf.register("stringLengthInt", lambda x: len(x), IntegerType())
>>> sqlContext.sql("SELECT stringLengthInt('test')").collect()
[Row(stringLengthInt(test)=4)]

#Using sqlContext.
>>> sqlContext.registerFunction("stringLengthString", lambda x: len(x))
>>> sqlContext.sql("SELECT stringLengthString('test')").collect()
[Row(stringLengthString(test)=u'4')]

>>> from pyspark.sql.types import IntegerType
>>> sqlContext.registerFunction("stringLengthInt", lambda x: len(x), IntegerType())
>>> sqlContext.sql("SELECT stringLengthInt('test')").collect()
[Row(stringLengthInt(test)=4)]

#Use pyspark.sql.functions.udf(f=None, returnType=StringType)
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
slen = udf(lambda s: len(s), IntegerType())

@udf
def to_upper(s):
    if s is not None:
        return s.upper()

#for later version below decorator is possible 
#@udf(returnType=IntegerType())
def add_one(x):
    if x is not None:
        return x + 1
add_one = udf(add_one, IntegerType()) 

df = spark.createDataFrame([(1, "John Doe", 21)], ("id", "name", "age"))
>>> df.select(slen("name").alias("slen(name)"), to_upper("name"), add_one("age")).show()
+----------+--------------+------------+
|slen(name)|to_upper(name)|add_one(age)|
+----------+--------------+------------+
|         8|      JOHN DOE|          22|
+----------+--------------+------------+


#Using sqlContext.registerJavaFunction(name, javaClassName, returnType=None)
>>> sqlContext.registerJavaFunction("javaStringLength", "test.org.apache.spark.sql.JavaStringLength", IntegerType())
>>> sqlContext.sql("SELECT javaStringLength('test')").collect()
[Row(UDF(test)=4)]
>>> sqlContext.registerJavaFunction("javaStringLength2","test.org.apache.spark.sql.JavaStringLength")
>>> sqlContext.sql("SELECT javaStringLength2('test')").collect()
[Row(UDF(test)=4)]




##Spark - DataFrame - Getting a specific row or column of DF 


df.select(*cols)
    Projects a set of expressions and returns a new DataFrame.
    cols – list of column names (string) 
           or expressions involving Column instances (ie df.colName)
           If one of the column names is '*', 
           that column is expanded to include all columns in the current DataFrame. 
df.show(n=20, truncate=True)
    Prints the first n rows to the console.
df.columns
    Returns all column names as a list.
    >>> df.columns
    ['age', 'name']
df.collect()
    Returns all the records as a list of Row.
    >>> df.collect()
    [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
pyspark.sql.functions.col(colName), pyspark.sql.functions.column(colName)
    Returns a Column based on the given column name.    
    
#Getting a named column is 
df.select("colname")  #return DataFrame , to display, do .show()

#To get Single column from  column index 
from pyspark.sql.functions import col
df.select(col(df.columns[index])) #return DataFrame, to display, do .show()


#To get a row 
##Not straight forward as DF are distributed 
df = spark.createDataFrame([("a", 1), ("b", 2), ("c", 3)], ["letter", "name"])
myIndex = 1
values = (df.rdd.zipWithIndex()     #(k,v),index 
            .filter(lambda t,i : i == myIndex)
            .map(lambda t,i: t)
            .collect())
print(values[0])
# (u'b', 2)




##Spark - DataFrame -  Usage of Column and DF operations 
class Column 
    A column in a DataFrame.

#check functions at http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions
#check column operations http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column


#Creation of Column Instance 
#Note below operation  gives Column instance
#not  DataFrame with single column
#to convert to DataFrame, use as  df.select(Column_instance)
1. Select a Column instance out of a DataFrame
    df.colName
    df["colName"] 

2. Create from an expression
    df.colName + 1
    1 / df.colName

3. Using pyspark.sql.functions
    from pyspark.sql import functions as F
    F.func(col)  # where col is DF col eg df.colName or df["colName"]

#Example 
df = spark.createDataFrame([(1, "John Doe", 21)], ("id", "name", "age"))
>>> df.name
Column<b'name'>
>>> df.name + "OK"
Column<b'(name + OK)'>
>>> df.select(df.name)
DataFrame[name: string]



#complex Example 
from pyspark.sql import functions as F
peopleDF.select(F.when(peopleDF["name"] == "Michael", 0)
                 .when(peopleDF["name"] == "Andy", 1)
                 .otherwise(2).alias("cond"), peopleDF["age"]) #select two columns 
              .show()
  
##Spark - DataFrame - Column Reference 
df =  sc.parallelize([(2, 'Alice'), (5, 'Bob')]) \
        .toDF(StructType([StructField('age', IntegerType()),
                          StructField('name', StringType())]))

class pyspark.sql.Column(jc)
    #Attributes of column instance eg df.colName or df["colName"] or F.col("colName")
    Operators Possible 
        #boolean operator , elementwise 
            bitwiseAND(col2) ,col1 & col2 , col1 & literal for boolean and 
            bitwiseOR(col2) , col1 | col2, col1 | literal for boolean or 
            bitwiseXOR(col2) , ~col for boolean not 
        # arithmetic operators, elementwise 
            -col
            col1 + col2 , col1 + literal or literal + col1 
            col1 - col2 , col1 - literal or literal - col1 
            col1 * col2 , col1 * literal or literal * col1 
            col1 / col2 , col1 / literal or literal / col1 
            col1 // col2 , col1 // literal or literal // col1 
            col1 % col2 , col1 % literal or literal % col1 
            col1 ** col2 , col1 ** literal or literal ** col1 
        # logistic operators, elementwise 
            col1 == col2 , col1 == literal or literal == col1 
            col1 != col2 , col1 != literal or literal != col1 
            col1 >= col2 , col1 >= literal or literal >= col1 
            col1 > col2 , col1 > literal or literal > col1 
            col1 <= col2 , col1 <= literal or literal <= col1 
            col1 < col2 , col1 < literal or literal < col1         
    rlike(regex)
        Return a Boolean :class:`Column` based on a regex match.
        >>> df.filter(df.name.rlike('ice$')).collect()
        [Row(age=2, name=u'Alice')]  
    like(SQL_LIKE_pattern)
        Return a Boolean :class:`Column` based on a SQL LIKE match.
        >>> df.filter(df.name.like('Al%')).collect()
        [Row(age=2, name=u'Alice')]
    startswith(string)
        Return a Boolean Column based on a string match.
        >>> df.filter(df.name.startswith('Al')).collect()
        [Row(age=2, name=u'Alice')]
        >>> df.filter(df.name.startswith('^Al')).collect()
        []
    endswith(string)
        Return a Boolean Column based on matching end of string.
        >>> df.filter(df.name.endswith('ice')).collect()
        [Row(age=2, name=u'Alice')]
        >>> df.filter(df.name.endswith('ice$')).collect()
        []    
    isNull()
        True if the current expression is null. Often combined with
        DataFrame.filter to select rows with null values.
        >>> from pyspark.sql import Row
        >>> df2 = sc.parallelize([Row(name=u'Tom', height=80), Row(name=u'Alice', height=None)]).toDF()
        >>> df2.filter(df2.height.isNull()).collect()
        [Row(height=None, name=u'Alice')]
    isNotNull()
        True if the current expression is null. Often combined with
        DataFrame.filter to select rows with non-null values.
        >>> from pyspark.sql import Row
        >>> df2 = sc.parallelize([Row(name=u'Tom', height=80), Row(name=u'Alice', height=None)]).toDF()
        >>> df2.filter(df2.height.isNotNull()).collect()
        [Row(height=80, name=u'Tom')]
    alias(*alias),name(*alias)
        Returns this column aliased with a new name or names 
        (in the case of expressions that return more than one column, such as explode).
        >>> df.select(df.age.alias("age2")).collect()
        [Row(age2=2), Row(age2=5)]
    asc()
    desc()
        Returns a sort expression based on the ascending/descending order of the given column name.
    between(lowerBound, upperBound)
        A boolean expression that is evaluated to true 
        if the value of this expression is between the given columns.
        >>> df.select(df.name, df.age.between(2, 4)).show()
        +-----+---------------------------+
        | name|((age >= 2) AND (age <= 4))|
        +-----+---------------------------+
        |Alice|                       true|
        |  Bob|                      false|
        +-----+---------------------------+
    cast(dataType),astype(dataType)
        Convert the column into type dataType (dataType can be string of Datatype)
        >>> df.select(df.age.cast("string").alias('ages')).collect()
        [Row(ages=u'2'), Row(ages=u'5')]
        >>> df.select(df.age.cast(StringType()).alias('ages')).collect()
        [Row(ages=u'2'), Row(ages=u'5')]
    getField(name)
        An expression that gets a field by name in a StructField.
        >>> from pyspark.sql import Row
        >>> df = sc.parallelize([Row(r=Row(a=1, b="b"))]).toDF()
        >>> df.select(df.r.getField("b")).show()
        +---+
        |r.b|
        +---+
        |  b|
        +---+
        >>> df.select(df.r.a).show()
        +---+
        |r.a|
        +---+
        |  1|
        +---+
    getItem(key)
        An expression that gets an item at index position out of a list, 
        or gets an item by key out of a dict.
        >>> df = sc.parallelize([ ([1, 2], {"key": "value"}) ]).toDF(["l", "d"])
        >>> df.select(df.l.getItem(0), df.d.getItem("key")).show()
        +----+------+
        |l[0]|d[key]|
        +----+------+
        |   1| value|
        +----+------+
        >>> df.select(df.l[0], df.d["key"]).show()
        +----+------+
        |l[0]|d[key]|
        +----+------+
        |   1| value|
        +----+------+
    isNotNull()
        True if the current expression is not null.
    isNull()
        True if the current expression is null.
    isin(*cols)
        A boolean expression that is evaluated to true 
        if the value of this expression is contained by the evaluated values of the arguments.
        >>> df[df.name.isin("Bob", "Mike")].collect()
        [Row(age=5, name=u'Bob')]
        >>> df[df.age.isin([1, 2, 3])].collect()
        [Row(age=2, name=u'Alice')]
    over(window)
        Define a windowing column.
        >>> from pyspark.sql import Window
        >>> window = Window.partitionBy("name").orderBy("age").rowsBetween(-1, 1)
        >>> from pyspark.sql.functions import rank, min
        >>> df.select(rank().over(window), min('age').over(window))
    substr(startPos, length)
        Return a Column which is a substring of the column.
        >>> df.select(df.name.substr(1, 3).alias("col")).collect()
        [Row(col=u'Ali'), Row(col=u'Bob')]
    when(condition, value), otherwise(value)
        Evaluates a list of conditions 
        and returns one of multiple possible result expressions. 
        >>> from pyspark.sql import functions as F
        >>> df.select(df.name, F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0)).show()
        +-----+------------------------------------------------------------+
        | name|CASE WHEN (age > 4) THEN 1 WHEN (age < 3) THEN -1 ELSE 0 END|
        +-----+------------------------------------------------------------+
        |Alice|                                                          -1|
        |  Bob|                                                           1|
        +-----+------------------------------------------------------------+
    
 

  
##Spark - DataFrame - DF Reference 

df = sc.parallelize([(2, 'Alice'), (5, 'Bob')])\
        .toDF(StructType([StructField('age', IntegerType()),
                          StructField('name', StringType())]))
df2 = sc.parallelize([Row(name='Tom', height=80), Row(name='Bob', height=85)]).toDF()
df3 = sc.parallelize([Row(name='Alice', age=2),
                           Row(name='Bob', age=5)]).toDF()
df4 = sc.parallelize([Row(name='Alice', age=10, height=80),
                           Row(name='Bob', age=5, height=None),
                           Row(name='Tom', age=None, height=None),
                           Row(name=None, age=None, height=None)]).toDF()
sdf = sc.parallelize([Row(name='Tom', time=1479441846),
                           Row(name='Bob', time=1479442946)]).toDF()


class pyspark.sql.DataFrame(jdf, sql_ctx)
    A distributed collection of data grouped into named columns.
    A DataFrame is equivalent to a relational table in Spark SQL, 
    Create a DF from datasource or from spakSession.createDataFrame
    arg 'col' is given as 
        df.colName, df["colName"], "colName", df[F.col("colName")]
    arg 'colName' is always string 
    #Example 
        people = spark.read.parquet("...")
        department = spark.read.parquet("...")
        people.filter(people.age > 30) \
          .join(department, people.deptId == department.id) \
          .groupBy(department.name, "gender").agg({"salary": "avg", "age": "max"})      
    #Attributes 
    dtypes
        Returns all column names and their data types as a list.
        >>> df.dtypes
        [('age', 'int'), ('name', 'string')]
    explain(extended=False)
        Prints the (logical and physical) plans to the console for debugging purpose.
    schema
        Returns the schema of this DataFrame as a pyspark.sql.types.StructType.
        >>> df.schema
        StructType(List(StructField(age,IntegerType,true),StructField(name,StringType,true)))
    rdd
        Returns the content as an pyspark.RDD of Row.
    columns
        Returns all column names as a list.
        >>> df.columns
        ['age', 'name']
    collect()
        Returns all the records as a list of Row.
        >>> df.collect()
        [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
    count()
        Returns the number of rows in this DataFrame.
        >>> df.count()
        2
    distinct()
        Returns a new DataFrame containing the distinct rows in this DataFrame.
        >>> df.distinct().count()
        2
    first()
        Returns the first row as a Row.
        >>> df.first()
        Row(age=2, name=u'Alice')
    head(n=None)
        Returns the first n rows.
        >>> df.head()
        Row(age=2, name=u'Alice')
        >>> df.head(1)
        [Row(age=2, name=u'Alice')]
    limit(num)
        Limits the result count to the number specified.
        >>> df.limit(1).collect()
        [Row(age=2, name=u'Alice')]
        >>> df.limit(0).collect()
        []
   take(num)
        Returns the first num rows as a list of Row.
        >>> df.take(2)
        [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
    printSchema()
        Prints out the schema in the tree format.
        >>> df.printSchema()
        root
         |-- age: integer (nullable = true)
         |-- name: string (nullable = true)
    select(*cols)
        Projects a set of expressions and returns a new DataFrame.
        cols –  list of column names (string) 
                or expressions involving column instances ie df.colName etc 
                If one of the column names is '*', that column is expanded to include all columns in the current DataFrame. 
        >>> df.select('*').collect()
        [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
        >>> df.select('name', 'age').collect()
        [Row(name=u'Alice', age=2), Row(name=u'Bob', age=5)]
        >>> df.select(df.name, (df.age + 10).alias('age')).collect()
        [Row(name=u'Alice', age=12), Row(name=u'Bob', age=15)]
    selectExpr(*expr)
        Projects a set of SQL expressions and returns a new DataFrame.
        This is a variant of select() that accepts SQL expressions.
        (numeric can have +,-,*,/ or string can have || for concatenation)
        (comparison operators =, !=, > < >= <= )
        (check list of functions from https://docs.oracle.com/cd/B28359_01/server.111/b28286/functions001.htm#SQLRF51174 )
        >>> df.selectExpr("age * 2", "abs(age)").collect()
        [Row((age * 2)=4, abs(age)=2), Row((age * 2)=10, abs(age)=5)]
    show(n=20, truncate=True)
        Prints the first n rows to the console.
        >>> df
        DataFrame[age: int, name: string]
        >>> df.show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        |  5|  Bob|
        +---+-----+
        >>> df.show(truncate=3)
        +---+----+
        |age|name|
        +---+----+
        |  2| Ali|
        |  5| Bob|
        +---+----+
    where(condition) or filter(condition)
        Filters rows using the given condition.
        condition – a Column of types.BooleanType or a string of SQL expression. 
        >>> df.filter(df.age > 3).collect()
        [Row(age=5, name=u'Bob')]
        >>> df.where(df.age == 2).collect()
        [Row(age=2, name=u'Alice')]
        >>> df.filter("age > 3").collect()
        [Row(age=5, name=u'Bob')]
        >>> df.where("age = 2").collect()
        [Row(age=2, name=u'Alice')]
    foreach(f)
        Applies the f(row) function to all Row of this DataFrame.
        This is a shorthand for df.rdd.foreach().
        >>> def f(person): #row 
                print(person.name)
        >>> df.foreach(f)
    foreachPartition(f)
        Applies the f(row_iterator) function to each partition of this DataFrame.
        This a shorthand for df.rdd.foreachPartition().
        >>> def f(people):
                for person in people:
                    print(person.name)
        >>> df.foreachPartition(f)
    approxQuantile(col, probabilities, relativeError)
        Calculates the approximate quantiles of a numerical column of a DataFrame.
    corr(col1, col2, method=None)
        Calculates the correlation of two columns of a DataFrame as a double value. 
    cov(col1, col2)
        Calculate the sample covariance for the given columns, 
        specified by their names, as a double value. 
        DataFrame.cov() and DataFrameStatFunctions.cov() are aliases.
    crosstab(col1, col2)
        Computes a pair-wise frequency table of the given columns. 
    stat
        Returns a DataFrameStatFunctions for statistic functions.
    describe(*cols)
        Computes statistics for numeric and string columns.
        Return DataFrame 
        >>> df.describe(['age']).show() #DataFrame[summary: string, age: string]
        +-------+------------------+
        |summary|               age|
        +-------+------------------+
        |  count|                 2|
        |   mean|               3.5|
        | stddev|2.1213203435596424|
        |    min|                 2|
        |    max|                 5|
        +-------+------------------+
        >>> df.describe().show()  #'age','name' or ['age','name']
        +-------+------------------+-----+
        |summary|               age| name|
        +-------+------------------+-----+
        |  count|                 2|    2|
        |   mean|               3.5| null|
        | stddev|2.1213203435596424| null|
        |    min|                 2|Alice|
        |    max|                 5|  Bob|
        +-------+------------------+-----+
    toDF(*cols)
        Returns a new DataFrame that with new specified column names
        >>> df.toDF('f1', 'f2').collect()
        [Row(f1=2, f2=u'Alice'), Row(f1=5, f2=u'Bob')]
    alias(alias)
        Returns a new DataFrame with an alias set.
        >>> from pyspark.sql.functions import *
        >>> df_as1 = df.alias("df_as1")
        >>> df_as2 = df.alias("df_as2")
        >>> joined_df = df_as1.join(df_as2, col("df_as1.name") == col("df_as2.name"), 'inner')
        >>> joined_df.select("df_as1.name", "df_as2.name", "df_as2.age").collect()
        [Row(name=u'Bob', name=u'Bob', age=5), Row(name=u'Alice', name=u'Alice', age=2)]
    withColumn(colName, col)
        Returns a new DataFrame by adding a column 
        or replacing the existing column that has the same name.
        >>> df.withColumn('age2', df.age + 2).collect()
        [Row(age=2, name=u'Alice', age2=4), Row(age=5, name=u'Bob', age2=7)]
    withColumnRenamed(existing, new)
        Returns a new DataFrame by renaming an existing column. 
        This is a no-op if schema doesn't contain the given column name.
        >>> df.withColumnRenamed('age', 'age2').collect()
        [Row(age2=2, name=u'Alice'), Row(age2=5, name=u'Bob')]
    drop(*cols)
        Returns a new DataFrame that drops the specified column. 
        This is a no-op if schema doesn't contain the given column name(s).
        >>> df.drop('age').collect()
        [Row(name=u'Alice'), Row(name=u'Bob')]
        >>> df.drop(df.age).collect()
        [Row(name=u'Alice'), Row(name=u'Bob')]
        >>> df.join(df2, df.name == df2.name, 'inner').drop(df.name).collect()
        [Row(age=5, height=85, name=u'Bob')]
        >>> df.join(df2, df.name == df2.name, 'inner').drop(df2.name).collect()
        [Row(age=5, name=u'Bob', height=85)]
        >>> df.join(df2, 'name', 'inner').drop('age', 'height').collect()
        [Row(name=u'Bob')]
    dropDuplicates(subset=None) or  drop_duplicates(subset=None)
        Return a new DataFrame with duplicate rows removed, 
        optionally only considering certain columns.
        >>> from pyspark.sql import Row
        >>> df = sc.parallelize([ \
                Row(name='Alice', age=5, height=80), \
                Row(name='Alice', age=5, height=80), \
                Row(name='Alice', age=10, height=80)]).toDF()
        >>> df.dropDuplicates().show()
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        |  5|    80|Alice|
        | 10|    80|Alice|
        +---+------+-----+
        >>> df.dropDuplicates(['name', 'height']).show()
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        |  5|    80|Alice|
        +---+------+-----+
    na
        Returns a DataFrameNaFunctions for handling missing values.
    dropna(how='any', thresh=None, subset=None)
        Returns a new DataFrame omitting rows with null values. 
        DataFrame.dropna() and DataFrameNaFunctions.drop() are aliases 
        •how – 'any' or 'all'. 
         If 'any', drop a row if it contains any nulls. 
         If 'all', drop a row only if all its values are null.
        •thresh – int, default None If specified, drop rows that have less than thresh non-null values. This overwrites the how parameter.
        •subset – optional list of column names to consider.
        >>> df4.na.drop().show()
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        | 10|    80|Alice|
        +---+------+-----+
    fillna(value, subset=None)
        Replace null values, 
        alias for na.fill(). DataFrame.fillna() and DataFrameNaFunctions.fill() 
        •value – int, long, float, string, or dict. 
         Value to replace null values with. 
         If the value is a dict, then subset is ignored 
         and value must be a mapping from column name (string) to replacement value. 
         The replacement value must be an int, long, float, or string.
        •subset – optional list of column names to consider. 
         Columns specified in subset that do not have matching data type are ignored. 
        >>> df4.na.fill(50).show()
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        | 10|    80|Alice|
        |  5|    50|  Bob|
        | 50|    50|  Tom|
        | 50|    50| null|
        +---+------+-----+
        >>> df4.na.fill({'age': 50, 'name': 'unknown'}).show()
        +---+------+-------+
        |age|height|   name|
        +---+------+-------+
        | 10|    80|  Alice|
        |  5|  null|    Bob|
        | 50|  null|    Tom|
        | 50|  null|unknown|
        +---+------+-------+
    replace(to_replace, value, subset=None)
        Returns a new DataFrame replacing a value with another value. 
        DataFrame.replace() and DataFrameNaFunctions.replace() are aliases 
        •to_replace – int, long, float, string, or list. 
           Value to be replaced. 
           If this is list, then list of values to be replaced 
           If it is a dict, then 'value' arg is ignored 
           and to_replace must be a mapping from column name (string) 
           to replacement value. The value to be replaced must be an int, long, float, or string.
        •value – int, long, float, string, or list. 
            Value to use to replace holes. 
            The replacement value must be an int, long, float, or string. 
            If value is a list or tuple, value should be of the same length with to_replace.
            If value is a scalar and to_replace is a sequence, then value is used as a replacement for each item in to_replace.
        •subset – optional list of column names to consider. 
            Columns specified in subset that do not have matching data type are ignored. 
        >>> df4.na.replace(10, 20).show()
        +----+------+-----+
        | age|height| name|
        +----+------+-----+
        |  20|    80|Alice|
        |   5|  null|  Bob|
        |null|  null|  Tom|
        |null|  null| null|
        +----+------+-----+
        >>> df4.na.replace(['Alice', 'Bob'], ['A', 'B'], 'name').show()
        +----+------+----+
        | age|height|name|
        +----+------+----+
        |  10|    80|   A|
        |   5|  null|   B|
        |null|  null| Tom|
        |null|  null|null|
        +----+------+----+
    freqItems(cols, support=None)
        Finding frequent items for columns, possibly with false positives. 
        •cols – Names of the columns to calculate frequent items 
                for as a list or tuple of strings.
        •support – The frequency with which to consider an item 'frequent'. 
                   Default is 1%. The support must be greater than 1e-4.
    agg(*exprs)
        Aggregate on the entire DataFrame without groups 
        (shorthand for df.groupBy.agg()).
        >>> df.agg({"age": "max"}).collect()
        [Row(max(age)=5)]
        >>> from pyspark.sql import functions as F
        >>> df.agg(F.min(df.age)).collect()
        [Row(min(age)=2)]
    cube(*cols)
        Create a multi-dimensional cube for the current DataFrame 
        using the specified columns, so we can run aggregation on them.
        #create cartesian product of (null, all values from name) x (null, all values from age)
        #null means  any value of that column possible 
        >>> df.cube("name", df.age).count().orderBy("name", "age").show()
        +-----+----+-----+
        | name| age|count|
        +-----+----+-----+
        | null|null|    2|
        | null|   2|    1|
        | null|   5|    1|
        |Alice|null|    1|
        |Alice|   2|    1|
        |  Bob|null|    1|
        |  Bob|   5|    1|
        +-----+----+-----+
    rollup(*cols)
        Create a multi-dimensional rollup for the current DataFrame using the specified columns, 
        so we can run aggregation on them.
        #create cartesian product of (null, all values from name) x (null, all values from age)
        #null means  any value of that column possible 
        #and them remove rows with "null" of first column only keeping null,null row
        >>> df.rollup("name", df.age).count().orderBy("name", "age").show()
        +-----+----+-----+
        | name| age|count|
        +-----+----+-----+
        | null|null|    2|
        |Alice|null|    1|
        |Alice|   2|    1|
        |  Bob|null|    1|
        |  Bob|   5|    1|
        +-----+----+-----+
    groupBy(*cols) or  groupby(*cols)
        Groups the DataFrame using the specified columns, returns GroupedData
        groupby() is an alias for groupBy().
        cols – list of columns to group by. 
              Each element should be a column name (string) or an expression (Column). 
        >>> df.groupBy().avg().collect()
        [Row(avg(age)=3.5)]
        >>> sorted(df.groupBy('name').agg({'age': 'mean'}).collect())
        [Row(name=u'Alice', avg(age)=2.0), Row(name=u'Bob', avg(age)=5.0)]
        >>> sorted(df.groupBy(df.name).avg().collect())
        [Row(name=u'Alice', avg(age)=2.0), Row(name=u'Bob', avg(age)=5.0)]
        >>> sorted(df.groupBy(['name', df.age]).count().collect())
        [Row(name=u'Alice', age=2, count=1), Row(name=u'Bob', age=5, count=1)]
    orderBy(*cols, **kwargs)
        Returns a new DataFrame sorted by the specified column(s).
        •cols – list of Column or column names to sort by.
        •ascending – boolean or list of boolean (default True). 
                    Sort ascending vs. descending. 
                    Specify list for multiple sort orders. 
                    If a list is specified, length of the list must equal length of the cols.
        >>> df.sort(df.age.desc()).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> df.sort("age", ascending=False).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> df.orderBy(df.age.desc()).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> from pyspark.sql.functions import *
        >>> df.sort(asc("age")).collect()
        [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
        >>> df.orderBy(desc("age"), "name").collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> df.orderBy(["age", "name"], ascending=[0, 1]).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
    sort(*cols, **kwargs)
        Returns a new DataFrame sorted by the specified column(s).
        >>> df.sort(df.age.desc()).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> df.sort("age", ascending=False).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> df.orderBy(df.age.desc()).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> from pyspark.sql.functions import *
        >>> df.sort(asc("age")).collect()
        [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
        >>> df.orderBy(desc("age"), "name").collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
        >>> df.orderBy(["age", "name"], ascending=[0, 1]).collect()
        [Row(age=5, name=u'Bob'), Row(age=2, name=u'Alice')]
    sortWithinPartitions(*cols, **kwargs)
        Returns a new DataFrame with each partition sorted by the specified column(s).
        >>> df.sortWithinPartitions("age", ascending=False).show()
        +---+-----+
        |age| name|
        +---+-----+
        |  2|Alice|
        |  5|  Bob|
        +---+-----+
    crossJoin(other)
        Returns the cartesian product with another DataFrame.
            other: Right side of the cartesian product
        >>> df.select("age", "name").collect()
        [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
        >>> df2.select("name", "height").collect()
        [Row(name=u'Tom', height=80), Row(name=u'Bob', height=85)]
        >>> df.crossJoin(df2.select("height")).select("age", "name", "height").collect()
        [Row(age=2, name=u'Alice', height=80), Row(age=2, name=u'Alice', height=85),
         Row(age=5, name=u'Bob', height=80), Row(age=5, name=u'Bob', height=85)]
    join(other, on=None, how=None)
        Joins with another DataFrame, using the given join expression.
        •other – Right side of the join
        •on – a string for join column name, a list of string column names, 
              or a join expression involving Column ( eg using two Columns and ==, > etc comparison operator)
              or a list of join expression
              If on is a string or a list of strings indicating the name of the join column(s), 
              the column(s) must exist on both sides, and this performs an equi-join.
        •how – str, default 'inner'. 
               One of inner, outer, left_outer, right_outer, leftsemi.
        #a full outer join between df1 and df2.
        >>> df.join(df2, df.name == df2.name, 'outer').select(df.name, df2.height).collect()
        [Row(name=None, height=80), Row(name=u'Bob', height=85), Row(name=u'Alice', height=None)]
        >>> df.join(df2, 'name', 'outer').select('name', 'height').collect()
        [Row(name=u'Tom', height=80), Row(name=u'Bob', height=85), Row(name=u'Alice', height=None)]
        >>> cond = [df.name == df3.name, df.age == df3.age]
        >>> df.join(df3, cond, 'outer').select(df.name, df3.age).collect()
        [Row(name=u'Alice', age=2), Row(name=u'Bob', age=5)]
        >>> df.join(df2, 'name').select(df.name, df2.height).collect()
        [Row(name=u'Bob', height=85)]
        >>> df.join(df4, ['name', 'age']).select(df.name, df.age).collect()
        [Row(name=u'Bob', age=5)]
    randomSplit(weights, seed=None)
        Randomly splits this DataFrame with the provided weights.
        >>> splits = df4.randomSplit([1.0, 2.0], 24)
        >>> splits[0].count()
        1
        >>> splits[1].count()
        3
    sample(withReplacement, fraction, seed=None)
        Returns a sampled subset of this DataFrame.
        >>> df.sample(False, 0.5, 42).count()
        2
    sampleBy(col, fractions, seed=None)
        Returns a stratified sample without replacement based on the fraction 
        given on each stratum.
        >>> from pyspark.sql.functions import col
        >>> dataset = sqlContext.range(0, 100).select((col("id") % 3).alias("key"))
        >>> sampled = dataset.sampleBy("key", fractions={0: 0.1, 1: 0.2}, seed=0)
        >>> sampled.groupBy("key").count().orderBy("key").show()
        +---+-----+
        |key|count|
        +---+-----+
        |  0|    5|
        |  1|    9|
        +---+-----+
    intersect(other)
        Return a new DataFrame containing rows only 
        in both this frame and another frame.
    subtract(other)
        Return a new DataFrame containing rows in this frame but not in another frame.
        This is equivalent to EXCEPT in SQL.
    unionAll(other) or  union(other)
        Return a new DataFrame containing union of rows in this frame and another frame.
        This is equivalent to UNION ALL in SQL. 
        To do a SQL-style set union (that does de-duplication of elements), 
        use this function followed by a distinct.        
    createGlobalTempView(name)
        Creates a global temporary view with this DataFrame.
        Local Temporary views in Spark SQL are session-scoped 
        and will disappear if the session that creates it terminates. 
        To share among all sessions and keep alive 
        until the Spark application terminates, create a global temporary view. 
        Global temporary view is tied to a system preserved database global_temp, 
        and use the qualified name to refer it, e.g. SELECT * FROM global_temp.view    
        >>> df.createGlobalTempView("people")
        >>> df2 = spark.sql("select * from global_temp.people")
        >>> sorted(df.collect()) == sorted(df2.collect())
        True
        >>> df.createGlobalTempView("people")  
        Traceback (most recent call last):
        ...
        AnalysisException: u"Temporary table 'people' already exists;"
        >>> spark.catalog.dropGlobalTempView("people")
    createOrReplaceTempView(name)
        Creates or replaces a local temporary view with this DataFrame.
        >>> df.createOrReplaceTempView("people")
        >>> df2 = df.filter(df.age > 3)
        >>> df2.createOrReplaceTempView("people")
        >>> df3 = spark.sql("select * from people")
        >>> sorted(df3.collect()) == sorted(df2.collect())
        True
        >>> spark.catalog.dropTempView("people")
    createTempView(name)
        Creates a local temporary view with this DataFrame.
        >>> df.createTempView("people")
        >>> df2 = spark.sql("select * from people")
        >>> sorted(df.collect()) == sorted(df2.collect())
        True
        >>> df.createTempView("people")  
        Traceback (most recent call last):
        ...
        AnalysisException: u"Temporary table 'people' already exists;"
        >>> spark.catalog.dropTempView("people")
    registerTempTable(name)
        Depreacted
        Registers this RDD as a temporary table using the given name.
        >>> df.registerTempTable("people")
        >>> df2 = spark.sql("select * from people")
        >>> sorted(df.collect()) == sorted(df2.collect())
        True
        >>> spark.catalog.dropTempView("people")        
    toJSON(use_unicode=True)
        Converts a DataFrame into a RDD of string.
        Each row is turned into a JSON document as one element in the returned RDD.
        >>> df.toJSON().first()
        u'{"age":2,"name":"Alice"}'
    toLocalIterator()
        Returns an iterator that contains all of the rows in this DataFrame. 
        >>> list(df.toLocalIterator())
        [Row(age=2, name=u'Alice'), Row(age=5, name=u'Bob')]
    toPandas()
        Returns the contents of this DataFrame as Pandas pandas.DataFrame.
        >>> df.toPandas()  
           age   name
        0    2  Alice
        1    5    Bob
    cache()
        Persists the DataFrame with the default storage level (MEMORY_AND_DISK).
    unpersist(blocking=False)
        Marks the DataFrame as non-persistent, 
        and remove all blocks for it from memory and disk.
    persist(storageLevel=StorageLevel(True, True, False, False, 1))
        Sets the storage level to persist the contents of the DataFrame across operations after the first time it is computed. This can only be used to assign a new storage level if the DataFrame does not have a storage level set yet. If no storage level is specified defaults to (MEMORY_AND_DISK).
    storageLevel
        Get the DataFrame's current storage level.
        >>> df.storageLevel
        StorageLevel(False, False, False, False, 1)
        >>> df.cache().storageLevel
        StorageLevel(True, True, False, True, 1)
        >>> df2.persist(StorageLevel.DISK_ONLY_2).storageLevel
        StorageLevel(True, False, False, False, 2)
    withWatermark(eventTime, delayThreshold)
        Defines an event time watermark for this DataFrame. 
        A watermark tracks a point in time before which we assume no more late data is going to arrive.
    write
        Interface for saving the content of the non-streaming DataFrame out into external storage.
        Note to read use sparkSession.read
    writeStream
        Interface for saving the content of the streaming DataFrame out into external storage.
        Note to read use sparkSession.readStream
    checkpoint(eager=True)
        Returns a checkpointed version of this Dataset. 
        Checkpointing can be used to truncate the logical plan of this DataFrame, 
        which is especially useful in iterative algorithms 
        where the plan may grow exponentially. 
        It will be saved to files inside the checkpoint directory set 
        with SparkContext.setCheckpointDir().
    isLocal()
        Returns True if the collect() and take() methods can be run locally 
    isStreaming
        Returns true if this Dataset contains one or more sources 
        that continuously return data as it arrives.





##Spark - DataFrame - pyspark.sql.functions Reference 
df = sc.parallelize([Row(name='Alice', age=2), Row(name='Bob', age=5)]).toDF()

    
pyspark.sql.functions module
    A collections of builtin functions operates on Column
        Use as 
            from pyspark.sql import functions as F
            df.select(F.func(col),...)
        where col is DF col eg df.colName or df["colName"]
        or any column expression via Column.method()
        or any functions of pyspark.sql.functions as many functions return Column
        Note *cols means list of column names (string) or list of Column expressions
        or variable number of column names or Columns 
        
    #Aggregate function
    Use inside of agg of df.groupBy(...).agg(F.func(col),..) or df.agg(F.func(col),...)
    avg(col)
        returns the average of the values in a group.
        >>> df.agg(F.avg("age"), F.min("age")).show()
        +--------+--------+
        |avg(age)|min(age)|
        +--------+--------+
        |     3.5|       2|
        +--------+--------+
        >>> df.groupBy("name").agg(F.avg("age"), F.min("age")).show()
        +-----+--------+--------+
        | name|avg(age)|min(age)|
        +-----+--------+--------+
        |  Bob|     5.0|       5|
        |Alice|     2.0|       2|
        +-----+--------+--------+
    collect_list(col)
        returns a list of objects with duplicates.
    collect_set(col)
        returns a set of objects with duplicate elements eliminated.    
    count(col)
        returns the number of items in a group.
    countDistinct(col, *cols)
        Returns a new Column for distinct count of col or cols.
        >>> df.agg(countDistinct(df.age, df.name).alias('c')).collect()
        [Row(c=2)]
        >>> df.agg(countDistinct("age", "name").alias('c')).collect()
        [Row(c=2)]            
    first(col, ignorenulls=False)
        returns the first value in a group.
    grouping(col)
        indicates whether a specified column in a GROUP BY list is aggregated or not, 
        returns 1 for aggregated or 0 for not aggregated in the result set.
        >>> df.cube("name").agg(grouping("name"), sum("age")).orderBy("name").show()
        +-----+--------------+--------+
        | name|grouping(name)|sum(age)|
        +-----+--------------+--------+
        | null|             1|       7|
        |Alice|             0|       2|
        |  Bob|             0|       5|
        +-----+--------------+--------+
    grouping_id(*cols)
        returns the level of grouping
        >>> df.cube("name").agg(grouping_id(), sum("age")).orderBy("name").show()
        +-----+-------------+--------+
        | name|grouping_id()|sum(age)|
        +-----+-------------+--------+
        | null|            1|       7|
        |Alice|            0|       2|
        |  Bob|            0|       5|
        +-----+-------------+--------+
    kurtosis(col)
        returns the kurtosis of the values in a group.
    last(col, ignorenulls=False)
        returns the last value in a group.
    max(col)
        returns the maximum value of the expression in a group.
    mean(col)
        returns the average of the values in a group.
    min(col)
        returns the minimum value of the expression in a group.
    skewness(col)
        returns the skewness of the values in a group.
    stddev(col)
        returns the unbiased sample standard deviation of the expression in a group.
    stddev_pop(col)
        returns population standard deviation of the expression in a group.
    stddev_samp(col)
        returns the unbiased sample standard deviation of the expression in a group.
    sum(col)
        returns the sum of all values in the expression.
    sumDistinct(col)
        returns the sum of distinct values in the expression.
    var_pop(col)
        returns the population variance of the values in a group.
    var_samp(col)
        returns the unbiased variance of the values in a group.
    functions.variance(col)
        returns the population variance of the values in a group.    
            

    #Window function 
    To be used along with WindowSpec eg F.func(col).over(window)
    Note other functions also can be used as F.func(col).over(window) eg 'min', 'max'
        >>> from pyspark.sql import Window
        >>> # PARTITION BY country ORDER BY date RANGE BETWEEN 3 PRECEDING AND 3 FOLLOWING
        >>> window = Window.orderBy("date").partitionBy("country").rangeBetween(-3, 3)
        >>> from pyspark.sql.functions import rank, min
        >>> df.select(rank().over(window), min('age').over(window))
    cume_dist()
        returns the cumulative distribution of values within a window partition, 
        i.e. the fraction of rows that are below the current row.
    dense_rank()
        returns the rank of rows within a window partition, without any gaps.
    rank()
        returns the rank of rows within a window partition.
        The difference between rank and denseRank is that 
        denseRank leaves no gaps in ranking sequence when there are ties. 
        For example, if you were ranking a competition using denseRank 
        and had three people tie for second place, you would say that all three were in second place 
        and that the next person came in third.
        Rank would give  sequential numbers, making the person that came in third place 
        (after the ties) would register as coming in fifth.
    lag(col, count=1, default=None)
        returns the value that is offset rows before the current row, 
        and defaultValue if there is less than offset rows before the current row. 
        For example, an offset of one will return the previous row at any given point 
        in the window partition.
    lead(col, count=1, default=None)
        returns the value that is offset rows after the current row
    ntile(n)
        returns the ntile group id (from 1 to n inclusive) 
        in an ordered window partition. 
        For example, if n is 4, the first quarter of the rows will get value 1, 
        the second quarter will get 2, the third quarter will get 3, 
        and the last quarter will get 4.
    percent_rank()
        returns the relative rank (i.e. percentile) of rows within a window partition.
    row_number()
        returns a sequential number starting at 1 within a window partition.

    #Collection functions
    array(*cols)
        Creates a new array column.
        >>> df.select(array('age', 'age').alias("arr")).collect()
        [Row(arr=[2, 2]), Row(arr=[5, 5])]
        >>> df.select(array([df.age, df.age]).alias("arr")).collect()
        [Row(arr=[2, 2]), Row(arr=[5, 5])]
    array_contains(col, value)
        >>> df = spark.createDataFrame([(["a", "b", "c"],), ([],)], ['data'])
        >>> df.select(array_contains(df.data, "a")).collect()
        [Row(array_contains(data, a)=True), Row(array_contains(data, a)=False)]
    size(col)
        returns the length of the array or map stored in the column.
        >>> df = spark.createDataFrame([([1, 2, 3],),([1],),([],)], ['data'])
        >>> df.select(size(df.data)).collect()
        [Row(size(data)=3), Row(size(data)=1), Row(size(data)=0)]
    sort_array(col, asc=True)
        sorts the input array in ascending or descending order according to the natural ordering of the array elements.
        >>> df = spark.createDataFrame([([2, 1, 3],),([1],),([],)], ['data'])
        >>> df.select(sort_array(df.data).alias('r')).collect()
        [Row(r=[1, 2, 3]), Row(r=[1]), Row(r=[])]
        >>> df.select(sort_array(df.data, asc=False).alias('r')).collect()
        [Row(r=[3, 2, 1]), Row(r=[1]), Row(r=[])]          

    #Mathematical functions 
    abs(col)
        Computes the absolute value.
    acos(col)
    cbrt(col)
        Computes the cube-root of the given value.
    ceil(col)
        Computes the ceiling of the given value.
    floor(col)
        Computes the floor of the given value.
    asin(col)
    atan(col)
    atan2(col1, col2)
    cos(col)
    cosh(col)
    bitwiseNOT(col)
    exp(col)
        Computes the exponential of the given value.
    expm1(col)
        Computes the exponential of the given value minus one.
    bround(col, scale=0)
        Round the given value to scale decimal places using HALF_EVEN rounding mode if scale >= 0 or at integral part when scale < 0.
        >>> spark.createDataFrame([(2.5,)], ['a']).select(bround('a', 0).alias('r')).collect()
        [Row(r=2.0)]   
    log(arg1, arg2=None)
        Returns the first argument-based logarithm of the second argument.
    log10(col)
        Computes the logarithm of the given value in Base 10.
    log1p(col)
        Computes the natural logarithm of the given value plus one.
    log2(col)
        Returns the base-2 logarithm of the argument.
    pow(col1, col2)
        Returns the value of the first argument raised to the power of the second argument.
    rint(col)
        Returns the double value that is closest in value to the argument 
        and is equal to a mathematical integer.
    round(col, scale=0)
        Round the given value to scale decimal places using HALF_UP rounding mode 
    signum(col)
        Computes the signum of the given value.
    sin(col)
        Computes the sine of the given value.
    sinh(col)
        Computes the hyperbolic sine of the given value.        
    sqrt(col)
        Computes the square root of the specified float value. 
    tan(col)
        Computes the tangent of the given value.
    tanh(col)
        Computes the hyperbolic tangent of the given value.
    hypot(col1, col2)
        Computes sqrt(a^2 + b^2) without intermediate overflow or underflow.
    expr(str)
        Parses the expression string into the column that it represents
        >>> df.select(expr("length(name)")).collect()
        [Row(length(name)=5), Row(length(name)=3)]
    factorial(col)
        Computes the factorial of the given value.
        >>> df = spark.createDataFrame([(5,)], ['n'])
        >>> df.select(factorial(df.n).alias('f')).collect()
        [Row(f=120)]    
    isnan(col)
        An expression that returns true iff the column is NaN.
        >>> df = spark.createDataFrame([(1.0, float('nan')), (float('nan'), 2.0)], ("a", "b"))
        >>> df.select(isnan("a").alias("r1"), isnan(df.a).alias("r2")).collect()
        [Row(r1=False, r2=False), Row(r1=True, r2=True)]
    isnull(col)
        An expression that returns true iff the column is null.
        >>> df = spark.createDataFrame([(1, None), (None, 2)], ("a", "b"))
        >>> df.select(isnull("a").alias("r1"), isnull(df.a).alias("r2")).collect()
        [Row(r1=False, r2=False), Row(r1=True, r2=True)]
    shiftLeft(col, numBits)
        Shift the given value numBits left.
    shiftRight(col, numBits)
        (Signed) shift the given value numBits right.
    shiftRightUnsigned(col, numBits)
        Unsigned shift the given value numBits right.        
    nanvl(col1, col2)
        Returns col1 if it is not NaN, or col2 if col1 is NaN.
        >>> df = spark.createDataFrame([(1.0, float('nan')), (float('nan'), 2.0)], ("a", "b"))
        >>> df.select(nanvl("a", "b").alias("r1"), nanvl(df.a, df.b).alias("r2")).collect()
        [Row(r1=1.0, r2=1.0), Row(r1=2.0, r2=2.0)]        
   
    #Misc functions 
    asc(col),desc(col)
        Returns a sort expression based on the as/decending order of the given column name.
    col(col),column(col)
        Returns a Column based on the given column name.
    lit(literal)
        Creates a Column of literal value.
    greatest(*cols)
        Returns the greatest value of the list of column names, 
        skipping null values. 
        This function takes at least 2 parameters. 
        It will return null iff all parameters are null.
        >>> df = spark.createDataFrame([(1, 4, 3)], ['a', 'b', 'c'])
        >>> df.select(greatest(df.a, df.b, df.c).alias("greatest")).collect()
        [Row(greatest=4)]
    least(*cols)
        Returns the least value of the list of column names, skipping null values. 
        This function takes at least 2 parameters. It will return null iff all parameters are null.
        >>> df = spark.createDataFrame([(1, 4, 3)], ['a', 'b', 'c'])
        >>> df.select(least(df.a, df.b, df.c).alias("least")).collect()
        [Row(least=1)]
    coalesce(*cols)
        Returns the first non null column value across *cols 
        >>> cDf = spark.createDataFrame([(None, None), (1, None), (None, 2)], ("a", "b"))
        >>> cDf.show()
        +----+----+
        |   a|   b|
        +----+----+
        |null|null|
        |   1|null|
        |null|   2|
        +----+----+
        >>> cDf.select(coalesce(cDf["a"], cDf["b"])).show()
        +--------------+
        |coalesce(a, b)|
        +--------------+
        |          null|
        |             1|
        |             2|
        +--------------+
        >>> cDf.select('*', coalesce(cDf["a"], lit(0.0))).show()
        +----+----+----------------+
        |   a|   b|coalesce(a, 0.0)|
        +----+----+----------------+
        |null|null|             0.0|
        |   1|null|             1.0|
        |null|   2|             0.0|
        +----+----+----------------+
    create_map(*cols)
        Creates a new map column.
        >>> df.select(create_map('name', 'age').alias("map")).collect()
        [Row(map={u'Alice': 2}), Row(map={u'Bob': 5})]
        >>> df.select(create_map([df.name, df.age]).alias("map")).collect()
        [Row(map={u'Alice': 2}), Row(map={u'Bob': 5})]
    struct(*cols)
        Creates a new struct column.
        >>> df.select(struct('age', 'name').alias("struct")).collect()
        [Row(struct=Row(age=2, name=u'Alice')), Row(struct=Row(age=5, name=u'Bob'))]
        >>> df.select(struct([df.age, df.name]).alias("struct")).collect()
        [Row(struct=Row(age=2, name=u'Alice')), Row(struct=Row(age=5, name=u'Bob'))]
    input_file_name()
        Creates a string column for the file name of the current Spark task.
    broadcast(df)
        Marks a DataFrame as small enough for use in broadcast joins.  
    spark_partition_id()
        A column for partition ID.
        >>> df.repartition(1).select(spark_partition_id().alias("pid")).collect()
        [Row(pid=0), Row(pid=0)]   
        
    #String functions 
    concat(*cols)
        Concatenates multiple input string columns together into a single string column.
        >>> df = spark.createDataFrame([('abcd','123')], ['s', 'd'])
        >>> df.select(concat(df.s, df.d).alias('s')).collect()
        [Row(s=u'abcd123')]
    concat_ws(sep, *cols)
        Concatenates multiple input string columns together into a single string column,
        using the given separator.       
    decode(col, charset)
        Computes the first argument into a string from a binary 
        using the provided character set (one of 'US-ASCII', 'ISO-8859-1', 'UTF-8', 'UTF-16BE', 'UTF-16LE', 'UTF-16').
    encode(col, charset)
        Computes the first argument into a binary from a string 
        using the provided character set (one of 'US-ASCII', 'ISO-8859-1', 'UTF-8', 'UTF-16BE', 'UTF-16LE', 'UTF-16').
    format_number(col, d)
        Formats the number X to a format like '#,–#,–#.–', rounded to d decimal places, and returns the result as a string.
        >>> spark.createDataFrame([(5,)], ['a']).select(format_number('a', 4).alias('v')).collect()
        [Row(v=u'5.0000')]
    format_string(format, *cols)
        Formats the arguments in printf-style and returns the result as a string column.
        >>> df = spark.createDataFrame([(5, "hello")], ['a', 'b'])
        >>> df.select(format_string('%d %s', df.a, df.b).alias('v')).collect()
        [Row(v=u'5 hello')]        
    conv(col, fromBase, toBase)
        Convert a number in a string column from one base to another.
        >>> df = spark.createDataFrame([("010101",)], ['n'])
        >>> df.select(conv(df.n, 2, 16).alias('hex')).collect()
        [Row(hex=u'15')]      
    ascii(col)
        Computes the numeric value of the first character of the string column.
    initcap(col)
        Translate the first letter of each word to upper case in the sentence.
    instr(str, substr)
        Locate the position of the first occurrence of substr column in the given string. Returns null if either of the arguments are null.
        >>> df = spark.createDataFrame([('abcd',)], ['s',])
        >>> df.select(instr(df.s, 'b').alias('s')).collect()
        [Row(s=2)]
    length(col)
        Calculates the length of a string or binary expression.
        >>> spark.createDataFrame([('ABC',)], ['a']).select(length('a').alias('length')).collect()
        [Row(length=3)]
    levenshtein(left, right)
        Computes the Levenshtein distance of the two given strings.
        >>> df0 = spark.createDataFrame([('kitten', 'sitting',)], ['l', 'r'])
        >>> df0.select(levenshtein('l', 'r').alias('d')).collect()
        [Row(d=3)]
    locate(substr, str, pos=1)
        Locate the position of the first occurrence of substr in a string column, after position pos.
        >>> df = spark.createDataFrame([('abcd',)], ['s',])
        >>> df.select(locate('b', df.s, 1).alias('s')).collect()
        [Row(s=2)]
    lower(col)
        Converts a string column to lower case.
    lpad(col, len, pad)
        Left-pad the string column to width len with pad.
    ltrim(col)
        Trim the spaces from left end for the specified string value.
    repeat(col, n)
        Repeats a string column n times, and returns it as a new string column.
        >>> df = spark.createDataFrame([('ab',)], ['s',])
        >>> df.select(repeat(df.s, 3).alias('s')).collect()
        [Row(s=u'ababab')]
    reverse(col)
        Reverses the string column and returns it as a new string column.
    rpad(col, len, pad)
        Right-pad the string column to width len with pad.
    rtrim(col)
        Trim the spaces from right end for the specified string value.
    substring(str, pos, len)
        Substring starts at pos and is of length len when str is String type or returns the slice of byte array that starts at pos in byte and is of length len when str is Binary type
        >>> df = spark.createDataFrame([('abcd',)], ['s',])
        >>> df.select(substring(df.s, 1, 2).alias('s')).collect()
        [Row(s=u'ab')]
    substring_index(str, delim, count)
        Returns the substring from string str before count occurrences of the delimiter delim. If count is positive, everything the left of the final delimiter (counting from left) is returned. If count is negative, every to the right of the final delimiter (counting from the right) is returned. substring_index performs a case-sensitive match when searching for delim.
        >>> df = spark.createDataFrame([('a.b.c.d',)], ['s'])
        >>> df.select(substring_index(df.s, '.', 2).alias('s')).collect()
        [Row(s=u'a.b')]
        >>> df.select(substring_index(df.s, '.', -3).alias('s')).collect()
        [Row(s=u'b.c.d')]
    translate(srcCol, matching, replace)
        A function translate any character in the srcCol by a character in matching. The characters in replace is corresponding to the characters in matching. The translate will happen when any character in the string matching with the character in the matching.
        >>> spark.createDataFrame([('translate',)], ['a']).select(translate('a', "rnlt", "123") \
        ...     .alias('r')).collect()
        [Row(r=u'1a2s3ae')]
    trim(col)
        Trim the spaces from both ends for the specified string column.
    upper(col)
        Converts a string column to upper case.
        
    #Regex functions 
    regexp_extract(col, pattern, idx)
            idx = particular group id to return 
        Extract a specific group matched by a Java regex, from the specified string column. 
        If the regex did not match, or the specified group did not match, an empty string is returned.
        >>> df = spark.createDataFrame([('100-200',)], ['str'])
        >>> df.select(regexp_extract('str', '(\d+)-(\d+)', 1).alias('d')).collect()
        [Row(d=u'100')]
        >>> df = spark.createDataFrame([('foo',)], ['str'])
        >>> df.select(regexp_extract('str', '(\d+)', 1).alias('d')).collect()
        [Row(d=u'')]
        >>> df = spark.createDataFrame([('aaaac',)], ['str'])
        >>> df.select(regexp_extract('str', '(a+)(b)?(c)', 2).alias('d')).collect()
        [Row(d=u'')]
    regexp_replace(col, pattern, replacement)
        Replace all substrings of the specified string value that match regexp with rep.
        >>> df = spark.createDataFrame([('100-200',)], ['str'])
        >>> df.select(regexp_replace('str', '(\d+)', '--').alias('d')).collect()
        [Row(d=u'-----')]
    split(col, pattern)
        Splits str around pattern (pattern is a regular expression).
        >>> df = spark.createDataFrame([('ab12cd',)], ['s',])
        >>> df.select(split(df.s, '[0-9]+').alias('s')).collect()
        [Row(s=[u'ab', u'cd'])]       

    #Date and time functions    
    col is datetime.date or datetime.datetime Column or string of colName     
    current_date()
        Returns the current date as a 'date' column.
    current_timestamp()
        Returns the current timestamp as a 'timestamp' column.        
    add_months(col, months)
        Returns the date that is months months after start
        >>> df = spark.createDataFrame([('2015-04-08',)], ['d'])
        >>> df.select(add_months(df.d, 1).alias('d')).collect()
        [Row(d=datetime.date(2015, 5, 8))]    
    date_add(col, days)
        Returns the date that is days days after start
        >>> df = spark.createDataFrame([('2015-04-08',)], ['d'])
        >>> df.select(date_add(df.d, 1).alias('d')).collect()
        [Row(d=datetime.date(2015, 4, 9))]
    date_format(col, format)
        Converts a date/timestamp/string to a value of string in the format specified 
        eg pattern- dd.MM.yyyy from java.text.SimpleDateFormat
        >>> df = spark.createDataFrame([('2015-04-08',)], ['a'])
        >>> df.select(date_format('a', 'MM/dd/yyy').alias('date')).collect()
        [Row(date=u'04/08/2015')]
    date_sub(col, days)
        Returns the date that is days days before start
        >>> df = spark.createDataFrame([('2015-04-08',)], ['d'])
        >>> df.select(date_sub(df.d, 1).alias('d')).collect()
        [Row(d=datetime.date(2015, 4, 7))]
    datediff(col_end, col_start)
        Returns the number of days from start to end.
        >>> df = spark.createDataFrame([('2015-04-08','2015-05-10')], ['d1', 'd2'])
        >>> df.select(datediff(df.d2, df.d1).alias('diff')).collect()
        [Row(diff=32)]
    dayofmonth(col)
        Extract the day of the month of a given date as integer.
        >>> df = spark.createDataFrame([('2015-04-08',)], ['a'])
        >>> df.select(dayofmonth('a').alias('day')).collect()
        [Row(day=8)]
    dayofyear(col)
        Extract the day of the year of a given date as integer.        
    hour(col)
        Extract the hours of a given date as integer.
    last_day(col)
        Returns the last day of the month which the given date belongs to.
        >>> df = spark.createDataFrame([('1997-02-10',)], ['d'])
        >>> df.select(last_day(df.d).alias('date')).collect()
        [Row(date=datetime.date(1997, 2, 28))]        
    minute(col)
        Extract the minutes of a given date as integer.
        >>> df = spark.createDataFrame([('2015-04-08 13:08:15',)], ['a'])
        >>> df.select(minute('a').alias('minute')).collect()
        [Row(minute=8)]
    month(col)
        Extract the month of a given date as integer.
        >>> df = spark.createDataFrame([('2015-04-08',)], ['a'])
        >>> df.select(month('a').alias('month')).collect()
        [Row(month=4)]
    months_between(col_date1, col_date2)
        Returns the number of months between date1 and date2.
        >>> df = spark.createDataFrame([('1997-02-28 10:30:00', '1996-10-30')], ['t', 'd'])
        >>> df.select(months_between(df.t, df.d).alias('months')).collect()
        [Row(months=3.9495967...)]
    next_day(col, dayOfWeek)
        Returns the first date which is later than the value of the date column.
        Day of the week parameter is case insensitive, and accepts:'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'.
        >>> df = spark.createDataFrame([('2015-07-27',)], ['d'])
        >>> df.select(next_day(df.d, 'Sun').alias('date')).collect()
        [Row(date=datetime.date(2015, 8, 2))]
    quarter(col)
        Extract the quarter of a given date as integer.
        >>> df = spark.createDataFrame([('2015-04-08',)], ['a'])
        >>> df.select(quarter('a').alias('quarter')).collect()
        [Row(quarter=2)]   
    second(col)
            Extract the seconds of a given date as integer.
    weekofyear(col)
        Extract the week number of a given date as integer.
        >>> df = spark.createDataFrame([('2015-04-08',)], ['a'])
        >>> df.select(weekofyear(df.a).alias('week')).collect()
        [Row(week=15)]
    year(col)
        Extract the year of a given date as integer.
        >>> df = spark.createDataFrame([('2015-04-08',)], ['a'])
        >>> df.select(year('a').alias('year')).collect()
        [Row(year=2015)]
    trunc(col, format)
        Returns date truncated to the unit specified by the format.
        format – 'year', 'YYYY', 'yy' or 'month', 'mon', 'mm' 
        >>> df = spark.createDataFrame([('1997-02-28',)], ['d'])
        >>> df.select(trunc(df.d, 'year').alias('year')).collect()
        [Row(year=datetime.date(1997, 1, 1))]
        >>> df.select(trunc(df.d, 'mon').alias('month')).collect()
        [Row(month=datetime.date(1997, 2, 1))] 
        
    #Conversion functions 
    to_date(col)
        Converts the column of pyspark.sql.types.StringType 
        or pyspark.sql.types.TimestampType into pyspark.sql.types.DateType.
        >>> df = spark.createDataFrame([('1997-02-28 10:30:00',)], ['t'])
        >>> df.select(to_date(df.t).alias('date')).collect()
        [Row(date=datetime.date(1997, 2, 28))]
    to_json(col, options={})
        Converts a column containing a [[StructType]] into a JSON string. 
        Throws an exception, in the case of an unsupported type.
        >>> from pyspark.sql import Row
        >>> from pyspark.sql.types import *
        >>> data = [(1, Row(name='Alice', age=2))]
        >>> df = spark.createDataFrame(data, ("key", "value"))
        >>> df.select(to_json(df.value).alias("json")).collect()
        [Row(json=u'{"age":2,"name":"Alice"}')]
    to_utc_timestamp(col, tz)
        Given a timestamp, which corresponds to a certain time of day 
        in the given timezone, returns another timestamp that corresponds 
        to the same time of day in UTC.
        >>> df = spark.createDataFrame([('1997-02-28 10:30:00',)], ['t'])
        >>> df.select(to_utc_timestamp(df.t, "PST").alias('t')).collect()
        [Row(t=datetime.datetime(1997, 2, 28, 18, 30))]    
    from_json(col, schema, options={})
        Parses a column containing a JSON string into a [[StructType]] with the specified schema. Returns null, in the case of an unparseable string.
        >>> from pyspark.sql.types import *
        >>> data = [(1, '''{"a": 1}''')]
        >>> schema = StructType([StructField("a", IntegerType())])
        >>> df = spark.createDataFrame(data, ("key", "value"))
        >>> df.select(from_json(df.value, schema).alias("json")).collect()
        [Row(json=Row(a=1))]
    from_unixtime(col_unixTimestamp, format='yyyy-MM-dd HH:mm:ss')
        Converts the number of seconds from unix epoch (1970-01-01 00:00:00 UTC) to a string representing the timestamp of that moment in the current system time zone in the given format.
    from_utc_timestamp(col_timestamp, tz)
        Given a timestamp, which corresponds to a certain time of day in UTC, returns another timestamp that corresponds to the same time of day in the given timezone.
    get_json_object(col, path)
        Extracts json object from a json string based on json path specified, 
        and returns json string of the extracted json object. 
        It will return null if the input json string is invalid.
        >>> data = [("1", '''{"f1": "value1", "f2": "value2"}'''), ("2", '''{"f1": "value12"}''')]
        >>> df = spark.createDataFrame(data, ("key", "jstring"))
        >>> df.select(df.key, get_json_object(df.jstring, '$.f1').alias("c0"), \
        ...                   get_json_object(df.jstring, '$.f2').alias("c1") ).collect()
        [Row(key=u'1', c0=u'value1', c1=u'value2'), Row(key=u'2', c0=u'value12', c1=None)]
    json_tuple(col, *fields)
        Creates a new row for a json column according to the given field names.
        >>> data = [("1", '''{"f1": "value1", "f2": "value2"}'''), ("2", '''{"f1": "value12"}''')]
        >>> df = spark.createDataFrame(data, ("key", "jstring"))
        >>> df.select(df.key, json_tuple(df.jstring, 'f1', 'f2')).collect()
        [Row(key=u'1', c0=u'value1', c1=u'value2'), Row(key=u'2', c0=u'value12', c1=None)]
    monotonically_increasing_id()
        A column that generates monotonically increasing 64-bit integers.
    degrees(col)
        Converts an angle measured in radians to an approximately equivalent angle measured in degrees.
    radians(col)
        Converts an angle measured in degrees to an approximately equivalent angle measured in radians.
    unbase64(col)
        Decodes a BASE64 encoded string column and returns it as a binary column.
    unhex(col)
        Inverse of hex. Interprets each pair of characters as a hexadecimal number and converts to the byte representation of number.
        >>> spark.createDataFrame([('414243',)], ['a']).select(unhex('a')).collect()
        [Row(unhex(a)=bytearray(b'ABC'))]
    unix_timestamp(col_timestamp, format='yyyy-MM-dd HH:mm:ss')
        Convert time string with given pattern ('yyyy-MM-dd HH:mm:ss', by default) 
        to Unix time stamp (in seconds), using the default timezone 
        and the default locale, return null if fail.

    #Advanced functions 
    udf(f, returnType=StringType)
        Creates a Column expression representing a user defined function (UDF) of f(col):returnType
        >>> from pyspark.sql.types import IntegerType
        >>> slen = udf(lambda s: len(s), IntegerType())
        >>> df.select(slen(df.name).alias('slen')).collect()
        [Row(slen=5), Row(slen=3)]
    when(condition, value)
            condition : Column with comparison operator 
            value : literal or Column expression
        Evaluates a list of conditions 
        and returns one of multiple possible result expressions. 
        If Column.otherwise() is not invoked, None is returned for unmatched conditions.
        >>> df.select(when(df['age'] == 2, 3).otherwise(4).alias("age")).collect()
        [Row(age=3), Row(age=4)]
        >>> df.select(when(df.age == 2, df.age + 1).alias("age")).collect()
        [Row(age=3), Row(age=None)]
    window(col_timestamp , windowDuration, slideDuration=None, startTime=None)
        Bucketize rows into one or more time windows given a column with timestamp 
        Window starts are inclusive but the window ends are exclusive, 
        e.g. 12:05 will be in the window [12:05,12:10) but not in [12:00,12:05). 
        Windows can support microsecond precision. 
        Windows in the order of months are not supported.

        The time column must be of pyspark.sql.types.TimestampType.
        Durations(windowDuration,slideDuration)  are provided as strings, 
        e.g. '1 second', '1 day 12 hours', '2 minutes'.
        Valid interval strings are 'week', 'day', 'hour', 'minute', 'second', 'millisecond', 'microsecond'. 
        If the slideDuration is not provided, the windows will be tumbling windows.

        The startTime is the offset with respect to 1970-01-01 00:00:00 UTC 
        with which to start window intervals. 

        The output column will be a struct called 'window' by default 
        with the nested columns 'start' and 'end', 
        where 'start' and 'end' will be of pyspark.sql.types.TimestampType.
        
        >>> df = spark.createDataFrame([("2016-03-11 09:00:07", 1)]).toDF("date", "val")
        >>> w = df.groupBy(window("date", "5 seconds")).agg(sum("val").alias("sum"))
        >>> w.select(w.window.start.cast("string").alias("start"),
                w.window.end.cast("string").alias("end"), "sum").collect()
        [Row(start=u'2016-03-11 09:00:05', end=u'2016-03-11 09:00:10', sum=1)]
    explode(col)
        Returns a new row for each element in the given col containing array or map.
        >>> from pyspark.sql import Row
        >>> eDF = spark.createDataFrame([Row(a=1, intlist=[1,2,3], mapfield={"a": "b"})])
        >>> eDF.select(explode(eDF.intlist).alias("anInt")).collect()
        [Row(anInt=1), Row(anInt=2), Row(anInt=3)]
        >>> eDF.select(explode(eDF.mapfield).alias("key", "value")).show()
        +---+-----+
        |key|value|
        +---+-----+
        |  a|    b|
        +---+-----+
    posexplode(col)
        Returns a new row for each element with position in the given col containing array or map.
        >>> from pyspark.sql import Row
        >>> eDF = spark.createDataFrame([Row(a=1, intlist=[1,2,3], mapfield={"a": "b"})])
        >>> eDF.select(posexplode(eDF.intlist)).collect()
        [Row(pos=0, col=1), Row(pos=1, col=2), Row(pos=2, col=3)]
        >>> eDF.select(posexplode(eDF.mapfield)).show()
        +---+---+-----+
        |pos|key|value|
        +---+---+-----+
        |  0|  a|    b|
        +---+---+-----+
        
    #Code creation functions
    crc32(col)
        Calculates the cyclic redundancy check value (CRC32) of a binary column and returns the value as a bigint.
    base64(col)
    bin(col)
        Returns the string representation of the binary value of the given column.
        >>> df.select(bin(df.age).alias('c')).collect()
        [Row(c=u'10'), Row(c=u'101')]     
    hash(*cols)
        Calculates the hash code of given columns, 
        and returns the result as an int column.
        >>> spark.createDataFrame([('ABC',)], ['a']).select(hash('a').alias('hash')).collect()
        [Row(hash=-757602832)]
    hex(col)
        Computes hex value of the given column, 
        >>> spark.createDataFrame([('ABC', 3)], ['a', 'b']).select(hex('a'), hex('b')).collect()
        [Row(hex(a)=u'414243', hex(b)=u'3')]
    md5(col)
        Calculates the MD5 digest 
    sha1(col)
        Returns the hex string result of SHA-1.
    sha2(col, numBits)
        Returns the hex string result of SHA-2 family of hash functions (SHA-224, SHA-256, SHA-384, and SHA-512)
    soundex(col)
        Returns the SoundEx encoding for a string
        
    #Statistical functions
    approxCountDistinct(col, rsd=None)
    approx_count_distinct(col, rsd=None)
    corr(col1, col2)
        Returns a new Column for the Pearson Correlation Coefficient for col1 and col2.
        >>> a = range(20)
        >>> b = [2 * x for x in range(20)]
        >>> df = spark.createDataFrame(zip(a, b), ["a", "b"])
        >>> df.agg(corr("a", "b").alias('c')).collect()
        [Row(c=1.0)]
    covar_pop(col1, col2)
        Returns a new Column for the population covariance of col1 and col2.
        >>> a = [1] * 10
        >>> b = [1] * 10
        >>> df = spark.createDataFrame(zip(a, b), ["a", "b"])
        >>> df.agg(covar_pop("a", "b").alias('c')).collect()
        [Row(c=0.0)]
    covar_samp(col1, col2)
        Returns new Column for sample covariance     
    rand(seed=None)
        Generates a random column with independent and identically distributed (i.i.d.) samples from U[0.0, 1.0].
    randn(seed=None)
        Generates a column with independent and identically distributed (i.i.d.) samples from the standard normal distribution.


        
        
###Spark - DataFrames - Special clasess - GroupedData, DataFrameNaFunctions, DataFrameStatFunctions, Window, WindowSpec
df = sc.parallelize([(2, 'Alice'), (5, 'Bob')]) \
    .toDF(StructType([StructField('age', IntegerType()),
                      StructField('name', StringType())]))
df3 = sc.parallelize([Row(name='Alice', age=2, height=80),
                           Row(name='Bob', age=5, height=85)]).toDF()
df4 = sc.parallelize([Row(course="dotNET", year=2012, earnings=10000),
                           Row(course="Java",   year=2012, earnings=20000),
                           Row(course="dotNET", year=2012, earnings=5000),
                           Row(course="dotNET", year=2013, earnings=48000),
                           Row(course="Java",   year=2013, earnings=30000)]).toDF()


                                   
class pyspark.sql.GroupedData(jgd, sql_ctx)
        Returned by df.groupBy(*cols)
        below functions return DataFrame, hence DataFrame methods can be used 
    agg(*exprs)
        Supported are avg, max, min, sum, count.
        exprs – a dict mapping from column name (string) to aggregate functions (string),  
                or a list of Column involving aggregation functions of pyspark.sql.function 
        >>> gdf = df.groupBy(df.name)
        >>> sorted(gdf.agg({"*": "count"}).collect())
        [Row(name=u'Alice', count(1)=1), Row(name=u'Bob', count(1)=1)]
        >>> from pyspark.sql import functions as F
        >>> sorted(gdf.agg(F.min(df.age)).collect())
        [Row(name=u'Alice', min(age)=2), Row(name=u'Bob', min(age)=5)]
    avg(*cols)
    mean(*cols)
        Computes average values for each numeric columns for each group.
        mean() is an alias for avg().
        >>> df.groupBy().avg('age').collect()
        [Row(avg(age)=3.5)]
        >>> df3.groupBy().avg('age', 'height').collect()
        [Row(avg(age)=3.5, avg(height)=82.5)]
    count()
        Counts the number of records for each group.
        >>> sorted(df.groupBy(df.age).count().collect())
        [Row(age=2, count=1), Row(age=5, count=1)]
    max(*cols)
    min(*cols)
        Computes the max/min value for each numeric columns for each group.
        >>> df.groupBy().max('age').collect()
        [Row(max(age)=5)]
        >>> df3.groupBy().max('age', 'height').collect()
        [Row(max(age)=5, max(height)=85)]
    sum(*cols)
        Compute the sum for each numeric columns for each group.
        >>> df.groupBy().sum('age').collect()
        [Row(sum(age)=7)]
        >>> df3.groupBy().sum('age', 'height').collect()
        [Row(sum(age)=7, sum(height)=165)]
    pivot(pivot_col, values=None)
        Returns GroupedData, hence call GroupedData aggregation method on retruns 
        It Performs this specified aggregation on each of 'values' of 'pivot_col'
        There are two versions of pivot function: 
         one that requires the caller to specify the list of distinct values to pivot on, 
         one that does not. 
        The latter is more concise but less efficient
        •pivot_col – Name of the column to pivot.
        •values – List of values that will be translated to columns 
                 in the output DataFrame.
        # Compute the sum of earnings for each year by course with each course as a separate column
        >>> df4.groupBy("year").pivot("course", ["dotNET", "Java"]).sum("earnings").collect()
        [Row(year=2012, dotNET=15000, Java=20000), Row(year=2013, dotNET=48000, Java=30000)]
        # Or without specifying column values (less efficient)
        >>> df4.groupBy("year").pivot("course").sum("earnings").collect()
        [Row(year=2012, Java=20000, dotNET=15000), Row(year=2013, Java=30000, dotNET=48000)]



##DataFrameNaFunctions
df = sc.parallelize([(2, 'Alice'), (5, 'Bob')])\
    .toDF(StructType([StructField('age', IntegerType()),
                      StructField('name', StringType())]))
df2 = sc.parallelize([Row(name='Tom', height=80), Row(name='Bob', height=85)]).toDF()
df3 = sc.parallelize([Row(name='Alice', age=2),
                               Row(name='Bob', age=5)]).toDF()
df4 = sc.parallelize([Row(name='Alice', age=10, height=80),
                               Row(name='Bob', age=5, height=None),
                               Row(name='Tom', age=None, height=None),
                               Row(name=None, age=None, height=None)]).toDF()
sdf = sc.parallelize([Row(name='Tom', time=1479441846),
                               Row(name='Bob', time=1479442946)]).toDF()


class pyspark.sql.DataFrameNaFunctions(df)
        Returned by df.na 
    drop(how='any', thresh=None, subset=None)
        Returns a new DataFrame omitting rows with null values. 
        DataFrame.dropna() and DataFrameNaFunctions.drop() are aliases of each other.
        >>> df4.na.drop().show()
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        | 10|    80|Alice|
        +---+------+-----+
    fill(value, subset=None)
        Replace null values, alias for na.fill(). 
        DataFrame.fillna() and DataFrameNaFunctions.fill() are aliases 
        >>> df4.na.fill(50).show()
        +---+------+-----+
        |age|height| name|
        +---+------+-----+
        | 10|    80|Alice|
        |  5|    50|  Bob|
        | 50|    50|  Tom|
        | 50|    50| null|
        +---+------+-----+
        >>> df4.na.fill({'age': 50, 'name': 'unknown'}).show()
        +---+------+-------+
        |age|height|   name|
        +---+------+-------+
        | 10|    80|  Alice|
        |  5|  null|    Bob|
        | 50|  null|    Tom|
        | 50|  null|unknown|
        +---+------+-------+
    replace(to_replace, value, subset=None)
        Returns a new DataFrame replacing a value with another value. 
        DataFrame.replace() and DataFrameNaFunctions.replace() are aliases 
        >>> df4.na.replace(10, 20).show()
        +----+------+-----+
        | age|height| name|
        +----+------+-----+
        |  20|    80|Alice|
        |   5|  null|  Bob|
        |null|  null|  Tom|
        |null|  null| null|
        +----+------+-----+
        >>> df4.na.replace(['Alice', 'Bob'], ['A', 'B'], 'name').show()
        +----+------+----+
        | age|height|name|
        +----+------+----+
        |  10|    80|   A|
        |   5|  null|   B|
        |null|  null| Tom|
        |null|  null|null|
        +----+------+----+

##DataFrameStatFunctions
df = sc.parallelize([(2, 'Alice'), (5, 'Bob')])\
        .toDF(StructType([StructField('age', IntegerType()),
                          StructField('name', StringType())]))
df2 = sc.parallelize([Row(name='Tom', height=80), Row(name='Bob', height=85)]).toDF()
df3 = sc.parallelize([Row(name='Alice', age=2),
                               Row(name='Bob', age=5)]).toDF()
df4 = sc.parallelize([Row(name='Alice', age=10, height=80),
                               Row(name='Bob', age=5, height=None),
                               Row(name='Tom', age=None, height=None),
                               Row(name=None, age=None, height=None)]).toDF()
sdf = sc.parallelize([Row(name='Tom', time=1479441846),
                               Row(name='Bob', time=1479442946)]).toDF()


class pyspark.sql.DataFrameStatFunctions(df)
        Returned by df.stat
    approxQuantile(col, probabilities, relativeError)
    corr(col1, col2, method=None)
    cov(col1, col2)
    crosstab(col1, col2)
    freqItems(cols, support=None)
    sampleBy(col, fractions, seed=None)
        >>> from pyspark.sql.functions import col
        >>> dataset = sqlContext.range(0, 100).select((col("id") % 3).alias("key"))
        >>> sampled = dataset.sampleBy("key", fractions={0: 0.1, 1: 0.2}, seed=0)
        >>> sampled.groupBy("key").count().orderBy("key").show()
        +---+-----+
        |key|count|
        +---+-----+
        |  0|    5|
        |  1|    9|
        +---+-----+


##Window
class pyspark.sql.Window
    Creation of WindowSpec to be used with Column.over(WindowSpec)
    All these classmethod returns Window, hence can be chained  
        >>> from pyspark.sql import Window
        >>> # ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        >>> window = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        >>> # PARTITION BY country ORDER BY date RANGE BETWEEN 3 PRECEDING AND 3 FOLLOWING
        >>> window = Window.orderBy("date").partitionBy("country").rangeBetween(-3, 3)
        >>> from pyspark.sql.functions import rank, min
        >>> df.select(rank().over(window), min('age').over(window))
    currentRow = 0
    classmethod  orderBy(*cols)
        Creates a WindowSpec with the ordering defined.        
    classmethod partitionBy(*cols)
        Creates a WindowSpec with the partitioning defined.
    classmethod rangeBetween(start, end)
        Creates a WindowSpec with the frame boundaries defined, 
        from start (inclusive) to end (inclusive).
        Both start and end are relative from the current row. 
        For example, '0' means 'current row', 
        while '-1' means one off before the current row, 
        and '5' means the five off after the current row.
    static rowsBetween(start, end)
        Creates a WindowSpec with the frame boundaries defined
    unboundedFollowing = 9223372036854775807
    LunboundedPreceding = -9223372036854775808L
    

class pyspark.sql.WindowSpec(jspec)
    Use the static methods in Window to create a WindowSpec.
    orderBy(*cols)
        Defines the ordering columns in a WindowSpec.
    partitionBy(*cols)
        Defines the partitioning columns in a WindowSpec.
    rangeBetween(start, end)
        Defines the frame boundaries, from start (inclusive) to end (inclusive).
    rowsBetween(start, end)
        Defines the frame boundaries, from start (inclusive) to end (inclusive).


        
        
   
###Spark - DataFrames - Example -  Basic creation and Operations on DF 
$ spark-submit  --master local[4] basic.py

#basic.py 
from __future__ import print_function

from pyspark.sql import *
from pyspark.sql.types import *

def basic_df_example(spark):
    df = spark.read.json(r"D:\Desktop\PPT\spark\data\people.json")
    # Displays the content of the DataFrame to stdout
    df.show()
    # +----+-------+
    # | age|   name|
    # +----+-------+
    # |null|Michael|
    # |  30|   Andy|
    # |  19| Justin|
    # +----+-------+

    # spark, df are from the previous example
    # Print the schema in a tree format
    df.printSchema()
    # root
    # |-- age: long (nullable = true)
    # |-- name: string (nullable = true)

    # Select only the "name" column
    df.select("name").show()
    # +-------+
    # |   name|
    # +-------+
    # |Michael|
    # |   Andy|
    # | Justin|
    # +-------+

    # Select everybody, but increment the age by 1
    df.select(df['name'], df['age'] + 1).show()
    # +-------+---------+
    # |   name|(age + 1)|
    # +-------+---------+
    # |Michael|     null|
    # |   Andy|       31|
    # | Justin|       20|
    # +-------+---------+

    # Select people older than 21
    df.filter(df['age'] > 21).show()
    # +---+----+
    # |age|name|
    # +---+----+
    # | 30|Andy|
    # +---+----+

    # Count people by age
    df.groupBy("age").count().show()
    # +----+-----+
    # | age|count|
    # +----+-----+
    # |  19|    1|
    # |null|    1|
    # |  30|    1|
    # +----+-----+

    # Register the DataFrame as a SQL temporary view
    df.createOrReplaceTempView("people")

    sqlDF = spark.sql("SELECT * FROM people")
    sqlDF.show()
    # +----+-------+
    # | age|   name|
    # +----+-------+
    # |null|Michael|
    # |  30|   Andy|
    # |  19| Justin|
    # +----+-------+

    # Register the DataFrame as a global temporary view
    df.createGlobalTempView("people")

    # Global temporary view is tied to a system preserved database `global_temp`
    spark.sql("SELECT * FROM global_temp.people").show()
    # +----+-------+
    # | age|   name|
    # +----+-------+
    # |null|Michael|
    # |  30|   Andy|
    # |  19| Justin|
    # +----+-------+

    # Global temporary view is cross-session
    spark.newSession().sql("SELECT * FROM global_temp.people").show()
    # +----+-------+
    # | age|   name|
    # +----+-------+
    # |null|Michael|
    # |  30|   Andy|
    # |  19| Justin|
    # +----+-------+



def schema_inference_example(spark):
    sc = spark.sparkContext

    # Load a text file and convert each line to a Row.
    lines = sc.textFile("examples/src/main/resources/people.txt")
    parts = lines.map(lambda l: l.split(","))
    people = parts.map(lambda p: Row(name=p[0], age=int(p[1])))

    # Infer the schema, and register the DataFrame as a table.
    schemaPeople = spark.createDataFrame(people)
    schemaPeople.createOrReplaceTempView("people")

    # SQL can be run over DataFrames that have been registered as a table.
    teenagers = spark.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")

    # The results of SQL queries are Dataframe objects.
    # rdd returns the content as an :class:`pyspark.RDD` of :class:`Row`.
    teenNames = teenagers.rdd.map(lambda p: "Name: " + p.name).collect()
    for name in teenNames:
        print(name)
    # Name: Justin



def programmatic_schema_example(spark):
    sc = spark.sparkContext
    # Load a text file and convert each line to a Row.
    lines = sc.textFile("examples/src/main/resources/people.txt")
    parts = lines.map(lambda l: l.split(","))
    # Each line is converted to a tuple.
    people = parts.map(lambda p: (p[0], p[1].strip()))

    # The schema is encoded in a string.
    schemaString = "name age"

    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
    schema = StructType(fields)

    # Apply the schema to the RDD.
    schemaPeople = spark.createDataFrame(people, schema)

    # Creates a temporary view using the DataFrame
    schemaPeople.createOrReplaceTempView("people")

    # Creates a temporary view using the DataFrame
    schemaPeople.createOrReplaceTempView("people")

    # SQL can be run over DataFrames that have been registered as a table.
    results = spark.sql("SELECT name FROM people")

    results.show()
    # +-------+
    # |   name|
    # +-------+
    # |Michael|
    # |   Andy|
    # | Justin|
    # +-------+


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    basic_df_example(spark)
    schema_inference_example(spark)
    programmatic_schema_example(spark)

    spark.stop()


    
    

    
        
    
    

    
/*****  DF - Data Sources  ****/
###Spark - DataFrame - Data Source 
class pyspark.sql.DataFrameReader(spark)
    Use sparkSession.read to access this.
    csv(path, schema=None, sep=None, encoding=None, quote=None, 
                escape=None, comment=None, header=None, inferSchema=None, 
                ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, 
                nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, 
                dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, 
                maxMalformedLogPerPartition=None, mode=None)
        Loads a CSV file and returns the result as a DataFrame.
        Automatically infer input schema if inferSchema is enabled. 
        or specify the schema explicitly using schema.
        >>> df = spark.read.csv('ages.csv')
        >>> df.dtypes
        [('_c0', 'string'), ('_c1', 'string')]
    format(source)
        Specifies the input data source format.
        source – string, name of the data source, e.g. 'json', 'parquet', 'jdbc', 'csv'
        >>> df = spark.read.format('json').load('people.json')
        >>> df.dtypes
        [('age', 'bigint'), ('name', 'string')]
    jdbc(url, table, column=None, lowerBound=None, upperBound=None, 
                numPartitions=None, predicates=None, properties=None)
        Construct a DataFrame representing the database table 
        •url – a JDBC URL of the form jdbc:subprotocol:subname
        •table – the name of the table
        •properties – a dictionary of JDBC database connection arguments. 
           For example { 'user' : 'SYSTEM', 'password' : 'mypassword' }
    json(path, schema=None, primitivesAsString=None, prefersDecimal=None, 
            allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, 
            allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, 
            mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None)
        Loads a JSON file (JSON Lines text format or newline-delimited JSON) 
        or an RDD of Strings storing JSON objects (one object per record) 
        and returns the result as a DataFrame
        >>> df1 = spark.read.json('people.json')
        >>> df1.dtypes
        [('age', 'bigint'), ('name', 'string')]
        >>> rdd = sc.textFile('python/test_support/sql/people.json')
        >>> df2 = spark.read.json(rdd)
        >>> df2.dtypes
        [('age', 'bigint'), ('name', 'string')]
    load(path=None, format=None, schema=None, **options)
        Loads data from a data source and returns it as DataFrame
        •path – optional string or a list of string for file-system backed data sources.
        •format – optional string for format of the data source. Default to 'parquet'.
        •schema – optional pyspark.sql.types.StructType for the input schema.
        •options – all other string options
        >>> df = spark.read.load('parquet_partitioned', opt1=True,
                opt2=1, opt3='str')
        >>> df.dtypes
        [('name', 'string'), ('year', 'int'), ('month', 'int'), ('day', 'int')]
        >>> df = spark.read.format('json').load(['people.json','people1.json'])
        >>> df.dtypes
        [('age', 'bigint'), ('aka', 'string'), ('name', 'string')]
    option(key, value)
        Adds an input option for the underlying data source.
    options(**options)
        Adds input options for the underlying data source.
    orc(path)
        Loads an ORC file, returning the result as a DataFrame.
        Currently ORC support is only available together with Hive support.
        >>> df = spark.read.orc('orc_partitioned')
        >>> df.dtypes
        [('a', 'bigint'), ('b', 'int'), ('c', 'int')]
    parquet(*paths)
        Loads a Parquet file, returning the result as a DataFrame.
        •mergeSchema: sets whether we should merge schemas collected from all Parquet part-files. 
        This will override spark.sql.parquet.mergeSchema. 
        The default value is specified in spark.sql.parquet.mergeSchema.
        >>> df = spark.read.parquet('parquet_partitioned')
        >>> df.dtypes
        [('name', 'string'), ('year', 'int'), ('month', 'int'), ('day', 'int')]
    schema(schema)
        Specifies the input schema.
        Some data sources (e.g. JSON) can infer the input schema automatically from data.
        schema – a pyspark.sql.types.StructType object 
    table(tableName)
        Returns the specified table as a DataFrame.
        tableName – string, name of the table. 
        >>> df = spark.read.parquet('parquet_partitioned')
        >>> df.createOrReplaceTempView('tmpTable')
        >>> spark.read.table('tmpTable').dtypes
        [('name', 'string'), ('year', 'int'), ('month', 'int'), ('day', 'int')]
    text(paths)
        Loads text files and returns a DataFrame 
        whose schema starts with a string column named 'value', 
        Each line in the text file is a new row in the resulting DataFrame.
        paths – string, or list of strings, for input path(s). 
        >>> df = spark.read.text('text-test.txt')
        >>> df.collect()
        [Row(value=u'hello'), Row(value=u'this')]


df = spark.read.parquet('parquet_partitioned')

class pyspark.sql.DataFrameWriter(df)
    Use df.write to access this.
    csv(path, mode=None, compression=None, sep=None, 
                quote=None, escape=None, header=None, nullValue=None, 
                escapeQuotes=None, quoteAll=None, dateFormat=None, 
                timestampFormat=None)
        Saves the content of the DataFrame in CSV format at the specified path.
            mode – specifies the behavior of the save operation when data already exists.
                'append': Append contents of this DataFrame to existing data.
                'overwrite': Overwrite existing data.
                'ignore': Silently ignore this operation if data already exists.
                'error' (default case): Throw an exception if data already exists.
        >>> df.write.csv(os.path.join(tempfile.mkdtemp(), 'data'))
    format(source)
        Specifies the underlying output data source.
        source – string, name of the data source, e.g. 'json', 'parquet', 'jdbc', 'csv'
        >>> df.write.format('json').save(os.path.join(tempfile.mkdtemp(), 'data'))
    insertInto(tableName, overwrite=False)
        Inserts the content of the DataFrame to the specified table.
    jdbc(url, table, mode=None, properties=None)
        Saves the content of the DataFrame to an external database table via JDBC.
        •url – a JDBC URL of the form jdbc:subprotocol:subname
        •table – Name of the table in the external database.
        •mode – specifies the behavior of the save operation when data already exists.
                'append': Append contents of this DataFrame to existing data.
                'overwrite': Overwrite existing data.
                'ignore': Silently ignore this operation if data already exists.
                'error' (default case): Throw an exception if data already exists.
        •properties – a dictionary of JDBC database connection arguments. 
               For example { 'user' : 'SYSTEM', 'password' : 'mypassword' }
    json(path, mode=None, compression=None, dateFormat=None, timestampFormat=None)
        Saves the content of the DataFrame in JSON format at the specified path.
        >>> df.write.json(os.path.join(tempfile.mkdtemp(), 'data'))
    mode(saveMode)
        Specifies the behavior when data or table already exists.
        Options include:
                'append': Append contents of this DataFrame to existing data.
                'overwrite': Overwrite existing data.
                'ignore': Silently ignore this operation if data already exists.
                'error' (default case): Throw an exception if data already exists.        >>> df.write.mode('append').parquet(os.path.join(tempfile.mkdtemp(), 'data'))
    option(key, value)
        Adds an output option for the underlying data source.
    options(**options)
        Adds output options for the underlying data source.
    orc(path, mode=None, partitionBy=None, compression=None)
        Saves the content of the DataFrame in ORC format at the specified path.
        >>> orc_df = spark.read.orc('python/test_support/sql/orc_partitioned')
        >>> orc_df.write.orc(os.path.join(tempfile.mkdtemp(), 'data'))
    parquet(path, mode=None, partitionBy=None, compression=None)
        Saves the content of the DataFrame in Parquet format at the specified path.
        >>> df.write.parquet(os.path.join(tempfile.mkdtemp(), 'data'))
    partitionBy(*cols)
        Partitions the output by the given columns on the file system.
        >>> df.write.partitionBy('year', 'month').parquet(os.path.join(tempfile.mkdtemp(), 'data'))
    save(path=None, format=None, mode=None, partitionBy=None, **options)
        Saves the contents of the DataFrame to a data source.
        •path – the path in a Hadoop supported file system
        •format – the format used to save
        •mode –specifies the behavior of the save operation when data already exists.
            ◦append: Append contents of this DataFrame to existing data.
            ◦overwrite: Overwrite existing data.
            ◦ignore: Silently ignore this operation if data already exists.
            ◦error (default case): Throw an exception if data already exists.
        •partitionBy – names of partitioning columns
        •options – all other string options
        >>> df.write.mode('append').parquet(os.path.join(tempfile.mkdtemp(), 'data'))
    saveAsTable(name, format=None, mode=None, partitionBy=None, **options)
        Saves the content of the DataFrame as the specified table.
        •name – the table name
        •format – the format used to save
        •mode – one of append, overwrite, error, ignore (default: error)
        •partitionBy – names of partitioning columns
        •options – all other string options
    text(path, compression=None)
        Saves the content of the DataFrame in a text file at the specified path.

#Note other that above , one can use RDD related methods
#Note DF can be converted to RDD by df.rdd 
#and RDD can be converted to DF by spark.createDataFrame(rdd)

#set options via option/options method or via csv, json method call 


##Spark - DataFrame - Data Source - Save Modes    
#scala/Java                      PY                         Meaning
SaveMode.ErrorIfExists (default) "error" (default)          When saving a DataFrame to a data source, if data already exists, an exception is expected to be thrown.  
SaveMode.Append                  "append"                   When saving a DataFrame to a data source, if data/table already exists, contents of the DataFrame are expected to be appended to existing data.  
SaveMode.Overwrite               "overwrite"                Overwrite mode means that when saving a DataFrame to a data source, if data/table already exists, existing data is expected to be overwritten by the contents of the DataFrame.  
SaveMode.Ignore                  "ignore"                   Ignore mode means that when saving a DataFrame to a data source, if data already exists, the save operation is expected to not save the contents of the DataFrame and to not change the existing data. This is similar to a CREATE TABLE IF NOT EXISTS in SQL.  
 
#Example 
df.write.mode('append').parquet(path)

    
    
##Spark - DataFrame - Data Source - Saving to Persistent Tables
#DataFrames can also be saved as persistent tables into Hive metastore 
df.write.saveAsTable(table_name) 
#and loaded by 
spark.read.table(table_name) 

#Notice existing Hive deployment is not necessary to use this feature. 
#Spark will create a default local Hive metastore (using Derby) for you
 
#By default saveAsTable will create a 'managed table', 
#meaning that the location of the data will be controlled by the metastore. 
#Managed tables will also have their data deleted automatically when a table is dropped.
   
df.write.mode("overwrite").saveAsTable("tablename")
df = df.read.table("tablename")
   
   
   
##Spark - DataFrame - Data Source - Generic Load/Save Functions
df = spark.read.load(r"D:\Desktop\PPT\spark\data/users.parquet")  #unicode escape because of backward-slash-and-u, so use /
df.select("name", "favorite_color").write.save("namesAndFavColors.parquet")
   
    
    
##Spark - DataFrame - Data Source - Manually Specifying Options   
df = spark.read.load("people.json", format="json")
df.select("name", "age").write.save("namesAndAges.parquet", format="parquet")  
    
    
##Spark - DataFrame - Data Source -  Run SQL on files directly  
df = spark.sql("SELECT * FROM parquet.`examples/src/main/resources/users.parquet`")   
    
    
    
    
###Spark - DataFrame - Data Source - Parquet Files
#Parquet is a columnar format that is supported by many other data processing systems
peopleDF = spark.read.json("examples/src/main/resources/people.json")

# DataFrames can be saved as Parquet files, maintaining the schema information.
peopleDF.write.parquet("people.parquet")

# Read in the Parquet file created above.
# Parquet files are self-describing so the schema is preserved.
# The result of loading a parquet file is also a DataFrame.
parquetFile = spark.read.parquet("people.parquet")

# Parquet files can also be used to create a temporary view and then used in SQL statements.
parquetFile.createOrReplaceTempView("parquetFile")
teenagers = spark.sql("SELECT name FROM parquetFile WHERE age >= 13 AND age <= 19")
teenagers.show()
# +------+
# |  name|
# +------+
# |Justin|
# +------+

  
  
  
##Spark - Parquet - Schema Merging
#Like ProtocolBuffer, Avro, and Thrift, Parquet also supports schema evolution. 
#Users can start with a simple schema, and gradually add more columns to the schema as needed. 


from pyspark.sql import Row

# spark is from the previous example.
# Create a simple DataFrame, stored into a partition directory
sc = spark.sparkContext

squaresDF = spark.createDataFrame(sc.parallelize(range(1, 6))
                                  .map(lambda i: Row(single=i, double=i ** 2)))
squaresDF.write.parquet("data/test_table/key=1")

# Create another DataFrame in a new partition directory,
# adding a new column and dropping an existing column
cubesDF = spark.createDataFrame(sc.parallelize(range(6, 11))
                                .map(lambda i: Row(single=i, triple=i ** 3)))
cubesDF.write.parquet("data/test_table/key=2")

# Read the partitioned table
mergedDF = spark.read.option("mergeSchema", "true").parquet("data/test_table")
mergedDF.printSchema()



##Spark - Parquet - Hive metastore Parquet table conversion
#When reading from and writing to Hive metastore Parquet tables, 
#Spark SQL will try to use its own Parquet support instead of Hive SerDe for better performance. 
#This behavior is controlled by the spark.sql.hive.convertMetastoreParquet configuration, and is turned on by default.


##Spark - DataFrame - Data Source - Metadata Refreshing
#Spark SQL caches Parquet metadata for better performance

# spark is an existing SparkSession
spark.catalog.refreshTable("my_table")



##Spark - Parquet - Configuration
#Configuration of Parquet can be done using the setConf method on SparkSession 
#or by running SET key=value commands using SQL.
#http://spark.apache.org/docs/latest/sql-programming-guide.html#configuration





###Spark - DataFrame - Data Source - JSON Datasets
# spark is from the previous example.
sc = spark.sparkContext

# A JSON dataset is pointed to by path.
# The path can be either a single text file or a directory storing text files
path = "examples/src/main/resources/people.json"
peopleDF = spark.read.json(path)

jsonStrings = ['{"name":"Yin","address":{"city":"Columbus","state":"Ohio"}}']
otherPeopleRDD = sc.parallelize(jsonStrings)
otherPeople = spark.read.json(otherPeopleRDD)
otherPeople.show()



###Spark - DataFrame - Data Source - JDBC To Other Databases
#This functionality should be preferred over using JdbcRDD
# --conf spark.executor.extraClassPath=postgresql-9.4.1207.jar if required in executor  
$ pyspark --driver-class-path postgresql-9.4.1207.jar --jars postgresql-9.4.1207.jar
$ spark-submit --driver-class-path postgresql-9.4.1207.jar --jars postgresql-9.4.1207.jar

$ spark-submit --driver-class-path mysql-connector-java-5.1.34.jar  --jars mysql-connector-java-5.1.34.jar
$ pyspark --driver-class-path mysql-connector-java-5.1.34.jar --jars mysql-connector-java-5.1.34.jar


# Note: JDBC loading and saving can be achieved via either the load/save or jdbc methods
# Loading data from a JDBC source
jdbcDF = spark.read \
    .format("jdbc") \
    .option("url", "jdbc:postgresql:dbserver") \
    .option("dbtable", "schema.tablename") \
    .option("user", "username") \
    .option("password", "password") \
    .load()

jdbcDF2 = spark.read \
    .jdbc("jdbc:postgresql:dbserver", "schema.tablename",
          properties={"user": "username", "password": "password"})

# Saving data to a JDBC source
jdbcDF.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql:dbserver") \
    .option("dbtable", "schema.tablename") \
    .option("user", "username") \
    .option("password", "password") \
    .save()

jdbcDF2.write \
    .jdbc("jdbc:postgresql:dbserver", "schema.tablename",
          properties={"user": "username", "password": "password"})
  

  
  
 
    
###Spark - DataFrame - Data Source - Hive Table 
#Spark SQL also supports reading and writing data stored in Apache Hive
#If Hive dependencies can be found on the classpath, Spark will load them automatically. 
#Note that these Hive dependencies must also be present on all of the worker nodes, 
#as they will need access to the Hive serialization and deserialization libraries (SerDes) in order to access data stored in Hive.

#Configuration of Hive : %SPARK_HOME%\conf\hive-site.xml

#Users who do not have an existing Hive deployment can still enable Hive support
#When not configured by the hive-site.xml, the context automatically creates metastore_db in the current directory 
#and creates a directory configured by spark.sql.warehouse.dir, which defaults to the directory spark-warehouse in the current directory that the Spark application is started



#in 2.0 
sparkSession.builder.enableHiveSupport()
#Enables Hive support, including connectivity to a persistent Hive metastore, 
#support for Hive serdes, and Hive user-defined functions.

#Example 
from os.path import expanduser, join

from pyspark.sql import SparkSession
from pyspark.sql import Row

# warehouse_location points to the default location for managed databases and tables
warehouse_location = 'spark-warehouse'

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL Hive integration example") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()

# spark is an existing SparkSession
spark.sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING)")
spark.sql("LOAD DATA LOCAL INPATH 'data/kv1.txt' INTO TABLE src")

# Queries are expressed in HiveQL
spark.sql("SELECT * FROM src").show()

##Spark- Hive -  Interacting with Different Versions of Hive Metastore
#http://spark.apache.org/docs/latest/sql-programming-guide.html#interacting-with-different-versions-of-hive-metastore


##Spark- Hive -  ERROR - Caused by: java.lang.IllegalArgumentException: java.net.URISyntaxException: Relative path in absolute URI: file:spark-warehouse
#Solution*
#spark.sql.warehouse.dir to some properly-referenced directory, 
#say file:///tmp/spark-warehouse that uses /// (triple slashes), it creates under c:/tmp
 
 
 
##Spark- Hive -  Installing Hadoop Hive 
#Download http://redrockdigimark.com/apachemirror/hive/hive-2.1.1/apache-hive-2.1.1-bin.tar.gz

#Untar to c:\hive 
#Set HIVE_HOME=c:\hive 
#Include PATH to c:\hive\bin 

#The Hive metastore service stores the metadata for Hive tables 
#and partitions in a relational database, and provides clients (including Hive) access 
#to this information using the metastore service API. Use MySql for storing metadata

#Download mysql-connector-java-5.0.5.jar file and copy the jar file to %HIVE_HOME%/lib 

#Go to MySQL command-line and execute the below commands 
# we launch the MySQL daemon via (CYGWIN)
$ /usr/bin/mysqld_safe &
#shutdown
mysqladmin.exe -h 127.0.0.1 -u root   --connect-timeout=5 shutdown

$ mysql -u root -p
Enter password:
mysql> CREATE DATABASE metastore_db;

#Create a User [hiveuser] in MySQL database using root user. 
# hiveuser as 'userhive' and hivepassword as 'hivepwd'

mysql> CREATE USER 'userhive'@'%' IDENTIFIED BY 'hivepwd';
mysql> GRANT all on *.* to 'userhive'@localhost identified by 'hivepwd';

mysql> flush privileges;


#Configure the Metastore Service to communicate with the MySQL Database.
#conf/hive-site.xml
<configuration>
    <property>
        <name>javax.jdo.option.ConnectionURL</name>
        <value>jdbc:mysql://localhost:3306/metastore_db?createDatabaseIfNotExist=true</value>
        <description>metadata is stored in a MySQL server</description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionDriverName</name>
        <value>com.mysql.jdbc.Driver</value>
        <description>MySQL JDBC driver class</description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionUserName</name>
        <value>userhive</value>
        <description>user name for connecting to mysql server </description>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionPassword</name>
        <value>hivepwd</value>
        <description>password for connecting to mysql server </description>
    </property>
    <property>
        <name>datanucleus.schema.autoCreateAll</name>
        <value>true</value>
     </property>
     <property>
        <name>datanucleus.schema.autoCreateTables</name>
        <value>true</value>
     </property>
     <property>
        <name>hive.metastore.schema.verification</name>
        <value>false</value>
     </property>    
</configuration>

#hive needs both dfs and yarn
$ start-dfs & start-yarn 
$ hive 
hive> create external table studentpq3 (id STRING, name STRING, phone STRING, email STRING) STORED AS PARQUET;

##For Hive 0.10 - 0.12, 
#copy parquet-hadoop-1.8.1.jar, parquet-hive-bundle-1.6.0.jar to %HIVE_HOME%/lib 
create external table studentpq2 (id STRING, name STRING, phone STRING, email STRING)
  ROW FORMAT SERDE 'parquet.hive.serde.ParquetHiveSerDe'
  STORED AS 
    INPUTFORMAT "parquet.hive.DeprecatedParquetInputFormat"
    OUTPUTFORMAT "parquet.hive.DeprecatedParquetOutputFormat";
##For Hive 0.13 and later - native support  
create external table studentpq3 (id STRING, name STRING, phone STRING, email STRING) STORED AS PARQUET;


#Verifying HIVE installation
hive> show tables;
OK
Time taken: 2.798 seconds
hive> select * from studentpq3;
hive> select count(*) from studentpq3;

#MySql console:
$ mysql -u userhive -phivepwd metastore_db
mysql> show tables;


##Spark - Hive - Using spark-hadoop-hive 
#Start 
hadoop-dfs & hadoop-yarn 

#copy c:\hive\conf\hive-site.xml to c:\spark\conf  
#Note rename it after use else spark-shell fails 

#after running below code, check in hive 
hive> select count(*) from studentpq3;
100 

##Code 
from __future__ import print_function
#uses mysql for hive 
# spark-submit  --driver-class-path mysql-connector-java-5.1.34.jar  --conf spark.executor.extraClassPath=mysql-connector-java-5.1.34.jar  --jars ../mysql-connector-java-5.1.34.jar --master local[4] hive2.py

from os.path import expanduser, join

from pyspark.sql import SparkSession
from pyspark.sql import Row


"""
A simple example demonstrating Spark SQL Hive integration.

"""


if __name__ == "__main__":

    # warehouse_location points to the default location for managed databases and tables
    warehouse_location = "file:///tmp/spark-warehouse"

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Hive integration example") \
        .config("spark.sql.warehouse.dir", warehouse_location) \
        .enableHiveSupport() \
        .getOrCreate()

    # spark is an existing SparkSession
    students = spark.read.format("csv") \
                           .option("sep", "|") \
                           .option("header", True) \
                           .load(r"D:\Desktop\PPT\spark\data\StudentDataHeader.csv")
                           
    students.show()
    students.write.mode("overwrite").saveAsTable("studentpq3")




    # Queries are expressed in HiveQL
    spark.sql("SELECT * FROM studentpq3").show()

    # Aggregation queries are also supported.
    spark.sql("SELECT COUNT(*) FROM studentpq3").show()


    # The results of SQL queries are themselves DataFrames and support all normal functions.
    sqlDF = spark.sql("SELECT name, email FROM studentpq3 WHERE id < 10 ORDER BY name")

    sqlDF.show()


    # read
    studentDFrame = spark.read.table("studentpq3")

    studentDFrame.createOrReplaceTempView("studentsView")

    # Queries can then join DataFrame data with data stored in Hive.
    spark.sql("SELECT * FROM studentsView r JOIN studentpq3 s ON r.id = s.id").show()


    spark.stop()

 
 
###Spark - DataFrame - Data Source - Avro - data serialization system
#Can use backend as Hive 

#using command ine 
$ bin/spark-submit  --driver-class-path spark-avro_2.11-3.2.0.jar --packages com.databricks:spark-avro_2.11:3.2.0
$ pyspark  --driver-class-path spark-avro_2.11-3.2.0.jar --packages com.databricks:spark-avro_2.11:3.2.0
 
##Spark - Avro -   Avro -> Spark SQL conversion

#Avro type       Spark SQL type
boolean         BooleanType 
int             IntegerType 
long            LongType 
float           FloatType 
double          DoubleType 
bytes           BinaryType 
string          StringType 
record          StructType 
enum            StringType 
array           ArrayType 
map             MapType 
fixed           BinaryType 
union           1. union(int, long)  will be mapped to  LongType .
                2. union(float, double)  will be mapped to  DoubleType .
                3. union(something, null) , where  something  is any supported Avro type. 
                   This will be mapped to the same Spark SQL type as that of  something , with  nullable  set to  true .
                4. All other  union  types are considered complex. 
                   They will be mapped to  StructType  where field names are  member0 ,  member1 , etc., in accordance with members of the  union . 
                   This is consistent with the behavior when converting between Avro and Parquet.

##Spark - Avro -   Spark SQL -> Avro conversion
#Spark SQL type     Avro type
ByteType            int 
ShortType           int 
DecimalType         string 
BinaryType          bytes 
TimestampType       long 
StructType          record 



#Example schema - Json notation 
#user.avsc 
{"namespace": "example.avro",
 "type": "record",
 "name": "User",
 "fields": [
     {"name": "name", "type": "string"},
     {"name": "favorite_color", "type": ["string", "null"]},
     {"name": "favorite_numbers", "type": {"type": "array", "items": "int"}}
 ]
}



##Spark - Avro -   reading & and writing

# Creates a DataFrame from a specified directory
df = spark.read.format("com.databricks.spark.avro").load("data/episodes.avro")

subset = df.where("doctor > 5")
subset.write.format("com.databricks.spark.avro").save("data/output1")

#  Saves the subset of the Avro records read in
subset = df.where("name = 'Ben'")
subset.write.mode('overwrite').format("com.databricks.spark.avro").save(r"D:/Desktop/PPT/spark/data/output")
 
 
#with schema 
schema_rdd = spark.sparkContext.textFile(r"D:/Desktop/PPT/spark/data/user.avsc").collect()
s = "".join(schema_rdd)
df = spark.read.format("com.databricks.spark.avro").option("avroSchema", s).load(r"D:/Desktop/PPT/spark/data/users.avro")



###Spark - DataFrame - Data Source - libsvm
#org.apache.spark.ml.source.libsvm.LibSVMDataSource


#The loaded DataFrame has two columns: label containing labels stored as doubles 
#and features containing feature vectors stored as Vectors.
df = spark.read.format("libsvm")
  .option("numFeatures", "780")
  .load("data/sample_libsvm_data.txt")


#LIBSVM data source supports the following options:
"numFeatures": number of features. If unspecified or nonpositive, the number of features will be determined automatically at the cost of one additional pass. This is also useful when the dataset is already split into multiple files and you want to load them separately, because some features may not present in certain files, which leads to inconsistent feature dimensions.
"vectorType": feature vector type, "sparse" (default) or "dense". 








 
##http://spark.apache.org/docs/latest/streaming-programming-guide.html
/***** Streaming  ****/
###Spark - DataFrame - Streaming 

#RDD of continuous data processed with Transformation function with certain window duration
#Data can be ingested from many sources like Kafka, Flume, Kinesis, or TCP sockets, 


 
##Spark - Streaming - Streaming context 

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext(master, appName)
#batchDuration=must be high enough to capture data 
#DStream window and slide duration must be multiple of batchDuration
ssc = StreamingContext(sc, 1) #batchDuration=1 sec 

##Spark - Streaming - Steps for programming 
0.Create StreamingContext
1.Define the input sources by creating input DStreams(Discretized Stream).
  Note DStream is nothing but one discrete RDD generated by datasource  
  Hence DStream: A continuous sequence of RDDs (of the same type)
2.Define the streaming computations by applying transformation 
  and output operations to DStreams
3.Start receiving data and processing it using streamingContext.start().
4.Wait for the processing to be stopped (manually or due to any error) 
  using streamingContext.awaitTermination().
5.The processing can be manually stopped using streamingContext.stop().

#Points to remember:
•Once a context has been started, no new streaming computations can be set up 
 or added to it.
•Once a context has been stopped, it cannot be restarted.
•Only one StreamingContext can be active in a JVM at the same time.
•stop() on StreamingContext also stops the SparkContext. 
 To stop only the StreamingContext, set the optional parameter of stop() called stopSparkContext to False.
•A SparkContext can be re-used to create multiple StreamingContexts, 
 as long as the previous StreamingContext is stopped 
 (without stopping the SparkContext) before the next StreamingContext is created.
 
##Check deployment guidelines in 
#https://spark.apache.org/docs/latest/streaming-programming-guide.html#deploying-applications 
 
##Spark - Streaming - StreamingContext Reference 

class pyspark.streaming.StreamingListener
        Derive from this class and implement below methods to handle corresponding events 
        then call sparkStreamingContext.addStreamingListener(streamingListener)
    onBatchCompleted(batchCompleted)
        Called when processing of a batch of jobs has completed.
    onBatchStarted(batchStarted)
        Called when processing of a batch of jobs has started.
    onBatchSubmitted(batchSubmitted)
        Called when a batch of jobs has been submitted for processing.
    onOutputOperationCompleted(outputOperationCompleted)
        Called when processing of a job of a batch has completed
    onOutputOperationStarted(outputOperationStarted)
        Called when processing of a job of a batch has started.
    onReceiverError(receiverError)
        Called when a receiver has reported an error
    onReceiverStarted(receiverStarted)
        Called when a receiver has been started
    onReceiverStopped(receiverStopped)
        Called when a receiver has been stopped
        
        
class pyspark.streaming.StreamingContext(sparkContext, batchDuration=None, jssc=None)
        batchDuration:  the time interval (in seconds) at which streaming
                        data will be divided into batches
    addStreamingListener(streamingListener)
        Add a StreamingListenerobject for receiving system events related to streaming.
    classmethod getActive()
        Return either the currently active StreamingContext 
        (i.e., if there is a context started but not stopped) or None.
    classmethod getActiveOrCreate(checkpointPath, setupFunc)
        Either return the active StreamingContext 
        (i.e. currently started but not stopped), 
        or recreate a StreamingContext from checkpoint data 
        or create a new StreamingContext using the provided setupFunc function. 
        If the checkpointPath is None or does not contain valid checkpoint data, 
        then setupFunc() will be called to create a new context and setup DStreams.
    classmethod getOrCreate(checkpointPath, setupFunc)
        Either recreate a StreamingContext from checkpointPath 
        or if checkpointPath is empty, create a new StreamingContext by 
            ssc = setupFunc()
            ssc.checkpoint(checkpointPath) #set checkpoint dir 
    checkpoint(checkpointPath)
        Sets the context to periodically checkpoint the DStream operations 
        for master fault-tolerance. 
        The graph will be checkpointed every batch interval.
    awaitTermination(timeout=None)
        Wait for the execution to stop.
        timeout – time to wait in seconds 
    awaitTerminationOrTimeout(timeout)
        Wait for the execution to stop. Return true if it's stopped; 
        or throw the reported error during the execution; 
        or false if the waiting time elapsed before returning from the method.
    binaryRecordsStream(directory, recordLength)
            recordLength – Length of each record in bytes
        Create an input stream that monitors a Hadoop-compatible file system(local file or HDFS etc) 
        for new files and reads them as flat binary files with records of fixed length. 
        Files must be written to the monitored directory by 'moving' them 
        from another location within the same file system. 
        File names starting with . are ignored.
        Returns DStream of bytes of fixed length 
    textFileStream(directory)
        Create an input stream that monitors a Hadoop-compatible file system (local file or HDFS etc) 
        for new files and reads them as text files. 
        Files must be writen to the monitored directory by 'moving' them 
        from another location within the same file system. 
        File names starting with . are ignored.
        Returns Dstream of string (each line)
    queueStream(rdds, oneAtATime=True, default=None)
            oneAtATime: pick one rdd each time or pick all of them once.
            default:    The default rdd if no more in rdds
        Create an input Dstream from  list of RDDs or list of lists 
        Changes to the queue after the stream is created will not be recognized.
        queueStream doesn't support checkpointing
    socketTextStream(hostname, port, storageLevel=StorageLevel(True, True, False, False, 2))
        Create an input from TCP source hostname:port. 
        Data is received using a TCP socket 
        and receive byte is interpreted as UTF8 encoded \n delimited lines.
        Returns DStream of string (each line) 
    sparkContext
        Return SparkContext which is associated with this StreamingContext.
    start()
        Start the execution of the streams.
    stop(stopSparkContext=True, stopGraceFully=False)
        Stop the execution of the streams
    remember(duration)
        Set each DStreams in this context to remember RDDs 
        it generated in the last given duration. 
        DStreams remember RDDs only for a limited duration of time 
        and releases them for garbage collection. 
        duration – Minimum duration (in seconds) that each DStream should remember its RDDs 
    transform(dstreams, transformFunc)
        Create a new DStream by applying a function transformFunc
        The order of output is same as in input dstreams
            dstreams : list of DStream 
            transformFunc(list_of_rdds):Returns transformed list of rdds 
                Each RDD is from input DStream 
    union(*dstreams)
        Create a unified DStream from multiple DStreams of the same type 
        and same slide duration.
        Takes variable number of DStream of same type and same slide duration 

##Spark - Streaming - StreamingContext Reference 
#DStream[T] means backing RDD of type T 
#DStream[(K,V)] means backing RDD is of tuple of two elements K,V, K is called key 
#Note That type T,U,K,V,S etc can be any type even a tuple, list etc 
class pyspark.streaming.DStream(jdstream, ssc, jrdd_deserializer)
        A continuous sequence of RDDs (of the same type) 
        Note Transformation functions return DStream 
        which can be printed by .pprint()        
        Note functionality of below methods are similar to RDD 
        Only has new dynamics called Window(time duration) for aggregation 
        Parameters of few methods 
            •windowDuration – width of the window; must be a multiple of this DStream's batching interval
            •slideDuration – sliding interval of the window (i.e., the interval after which the new DStream will generate RDDs); must be a multiple of this DStream's batching interval
            •numPartitions – number of partitions of each RDD in the new DStream.
            •invFunc can be None, then it will reduce all the RDDs in a window, 
             could be slower than having invFunc.
             Note The reduced value of over a new window is calculated 
             using the old window's reduce value :
                 1.reduce the new values that entered the window (e.g., adding new counts)
                 2.'inverse reduce' the old values that left the window (e.g., subtracting old counts)
    cache()
        Persist the RDDs of this DStream with the default storage level (MEMORY_ONLY).
    checkpoint(interval)
        Enable periodic checkpointing of RDDs of this DStream
        interval – time in seconds
    context()
        Return the StreamingContext associated with this DStream
    persist(storageLevel)
        Persist the RDDs of this DStream with the given storage level
    repartition(numPartitions)
        Return a new DStream with an increased or decreased level of parallelism.
    partitionBy(numPartitions, partitionFunc=<function portable_hash at 0x7fc35dbc8e60>)
        Return a copy of the DStream in which each RDD(K,V) are partitioned 
        using the specified partitioner.
        partitionFunc takes key only and returns partition index 
        
    pprint(num=10)
        Print the first num elements of each RDD generated in this DStream.    
    foreachRDD(func)
        Apply a function to each RDD in this DStream
        func can take either rdd or timestamp,rdd      
        
    count()
        Return a new DStream in which each RDD[count as Long] has a single element generated 
        by counting each RDD[T] of this DStream.
    countByValue()
        Return a new DStream in which each RDD[(T,count as Long)] contains 
        the counts of each distinct value in each RDD[T] of this DStream.
    countByValueAndWindow(windowDuration, slideDuration, numPartitions=None)
        Return a new DStream in which each RDD[(T,count as Long)] contains the count of distinct elements
        in RDD[T] in a sliding window over this DStream.
    countByWindow(windowDuration, slideDuration)
        Return a new DStream in which each RDD[count as Long] has a single element 
        generated by counting the number of elements in a window over this RDD[T] of DStream. 
    groupByKey(numPartitions=None)
        Return a new RDD[(K, list_ofV)] of DStream by applying groupByKey on each RDD[(K,V)] of DStream.
    groupByKeyAndWindow(windowDuration, slideDuration, numPartitions=None)
        Return a new RDD[(K, list_ofV)] of DStream 
        by applying groupByKey on each RDD[(K,V)] of DStream over a sliding window. 
        Similar to DStream.groupByKey(), but applies it over a sliding window.
    cogroup(other, numPartitions=None)
        Return a new RDD[(K, (list_ofV, list_ofW))] of DStream by applying 'cogroup' 
        between RDD[(K,V)] of this DStream and other[(K,W)] DStream.
    combineByKey(createCombiner, mergeValue, mergeCombiners, numPartitions=None)
        Return a new DStream[(K,C)] by applying combineByKey((V) => C, transforming first V to C) 
        and then mergeValue((C, V) => C, transforming C,V to C) 
        and then mergeCombiners((C, C) => C, mearging two C   to each RDD[(K,V)] of input DStream
        
    reduce(func)
        Return a new RDD[T] of DStream in which each RDD has a single element generated 
        by reducing func((T, T) => T) each RDD[T] of this DStream.
    reduceByKey(func, numPartitions=None)
        Return a new RDD[(K,V)] of DStream by applying func((V, V) => V) to each RDD[(K,V)] of input DStream .
    reduceByKeyAndWindow(func, invFunc, windowDuration, slideDuration=None, numPartitions=None, filterFunc=None)
        Return a new RDD[(K,V)] of DStream by applying incremental func((V, V) => V)
        over a sliding window to each RDD[(K,V)] of input DStream .
    reduceByWindow(reduceFunc, invReduceFunc, windowDuration, slideDuration)
        Return a newof DStream in which each RDD[T] has a single element generated 
        by reducing all elements in a sliding window over this RDD[T] of DStream
            reduceFunc: (T, T) => T, invReduceFunc: (T, T) => T
    updateStateByKey(updateFunc, numPartitions=None, initialRDD=None)
            updateFunc: (list_ofV_fromRecentRDD_for_a_K, old_S) => new_S
            initialRDD[(K,V)] to seed the update process 
        Return a new 'state' ,ie RDD[(K,S)] of new DStream 
        where the state for each key,K is updated by applying the updateFunc on RDD[(K,V)] of input DStream
        #Example 
        def updateFunction(newValues, runningCount): #(list_of_new_values, old_state)
            if runningCount is None:
                runningCount = 0
            return sum(newValues, runningCount)  # add the new values with the previous running count to get the new count
        runningCounts = pairs.updateStateByKey(updateFunction)
   
    filter(f)
            f: (T) => Boolean
        Return a new RDD[T] of DStream containing only the elements 
        that satisfy predicate on RDD[T} of DStream             
    flatMap(f, preservesPartitioning=False)
            f: (T) => list_ofU
        Return a new RDD[U] of DStream by applying a function to all elements of this DStream, 
        and then flattening the results on RDD[T} of DStream  
    flatMapValues(f)
            f: (V) => list_ofU
        Return a new RDD[(K,U)] of  DStream by applying a flatmap function to the value of each key-value pairs 
        in this RDD[(K,V)] of DStream without changing the key.
    map(f, preservesPartitioning=False)
            f: (T) => U
        Return a new RDD[U] of DStream by applying a function to each element of DStream[T].
    mapPartitions(f, preservesPartitioning=False)
            f: list_ofT_in_a_partition => list_ofU
        Return a new RDD[U] of DStream[ in which each RDD is generated 
        by applying mapPartitions() to each RDD[T] of this DStream.
    mapPartitionsWithIndex(f, preservesPartitioning=False)
            f:(index,list_ofT_in_a_partition) => list_ofU
        Return a new RDD[U] of DStream in which each RDD is generated 
        by applying mapPartitionsWithIndex() to each RDD[T] of this DStream.
    mapValues(f)
            f: V => U 
        Return a new RDD[U] of DStream by applying a map function 
        to the value of each key-value pairs in this RDD[(K,V)] of DStream without changing the key.
    transform(func)
            func: takes either rdd[T] or timestamp,rdd[T} and returns rdd[U]
        Return a new DStream in which each RDD[U} is generated 
        by applying a function on each RDD[T} of this DStream.
    transformWith(func, other, keepSerializer=False)
            func: either rdd_a[T],rdd_b[U] or (time, rdd_a[T], rdd_b[U]) and returns rdd[V]
        Return a new DStream in which each RDD[V] is generated 
        by applying a function on each RDD[T] of this DStream and 'other' RDD[U] of DStream.

    glom()
        Return a new DStream in which RDD[list_of_T] is generated by applying glom() 
        to RDD[T] of this DStream.
        glom means take each partition's values as ist_of_T
    join(other, numPartitions=None)
        Return a new RDD[(K, (V, W))] of DStream by applying 'join' 
        between RDD[(K,V)] of this DStream and other RDD[(K,W)] of DStream.        
            #Example 
            stream1 = ...
            stream2 = ...
            joinedStream = stream1.join(stream2)
            #window 
            windowedStream1 = stream1.window(20)
            windowedStream2 = stream2.window(60)
            joinedStream = windowedStream1.join(windowedStream2)  
            #joining a windowed stream with a dataset.
            rdd = ... # some RDD
            windowedStream = stream.window(20)
            joinedStream = windowedStream.transform(lambda rdd: rdd.join(rdd))
    fullOuterJoin(other, numPartitions=None)
        Return a new RDD[(K, (V_orNone, W_orNone))] of DStream by applying 'full outer join' 
        between RDD[(K,V)] of this DStream and other RDD[(K,W)] of DStream.       
    leftOuterJoin(other, numPartitions=None)
        Return a new RDD[(K, (V, W_orNone))] of DStream by applying 'left outer join' 
        between RDD[(K,V)] of this DStream and other RDD[(K,W)] of  DStream.
    rightOuterJoin(other, numPartitions=None)
        Return a new RDD[(K, (V_orNone, W))] of  DStream by applying 'right outer join' 
        between RDD[(K,V)] of this DStream and other RDD[(K,W)] of  DStream.

    slice(begin, end)
        Return all the RDDs between 'begin' to 'end' (both included)
        begin, end could be datetime.datetime() or unix_timestamp
    union(other)
        Return a new RD[T] of DStream by unifying data of another DStream[T] with this DStream[T]
        other – Another DStream having the same interval (i.e., slideDuration) as this DStream. 
        
    window(windowDuration, slideDuration=None)
        Return a new DStream in which each RDD[T} contains 
        all the elements in seen in a sliding window of time over this DStream[T].
    saveAsTextFiles(prefix, suffix=None)
        Save each RDD in this DStream as a text file, 
        using string representation of elements.
        
        

##Spark - Streaming -  using foreachRDD correctly 
#foreachRDD can be used to send data  to external systems

#Wrong usage 
#creating a connection object at the Spark driver, 
#then use it in a Spark worker to save records in the RDDs
#this requires 'connection'  serialized and sent from the driver to the worker
#Openining connection should be at Worker thread 
def sendRecord(rdd):
    connection = createNewConnection()  # executed at the driver
    rdd.foreach(lambda record: connection.send(record))
    connection.close()

dstream.foreachRDD(sendRecord)

#Further wrong usage 
#creating a new connection for every record
def sendRecord(record):
    connection = createNewConnection()
    connection.send(record)
    connection.close()

dstream.foreachRDD(lambda rdd: rdd.foreach(sendRecord))  

#Right usage 
#Use foreachPartition to send large data with one open Connection
def sendPartition(iter):
    connection = createNewConnection()
    for record in iter:
        connection.send(record)
    connection.close()

dstream.foreachRDD(lambda rdd: rdd.foreachPartition(sendPartition))  

#right usage - further optimized 
def sendPartition(iter):
    # ConnectionPool is a static, lazily initialized pool of connections
    connection = ConnectionPool.getConnection()
    for record in iter:
        connection.send(record)
    # return to the pool for future reuse
    ConnectionPool.returnConnection(connection)

dstream.foreachRDD(lambda rdd: rdd.foreachPartition(sendPartition))  
    
     


 
 
##Spark - Streaming - Output Operations on Streams
dstream.pprint() 
dstream.saveAsTextFiles(prefix, [suffix])  
    Save this DStream contents as text files. 
    The file name at each batch interval is generated 
    based on prefix and suffix: "prefix-TIME_IN_MS[.suffix]".     

dstream.foreachRDD(func)                   
    The most generic output operator that applies a function, func, 
    to each RDD generated from the stream.    
    func takes either rdd or (rdd,time)


##Spark - Streaming - Conversion of RDD to and from DStream 

#From RDD to DStream                  
sparkStreamingContext.queueStream(rdds, oneAtATime=True, default=None)
    rdds : List of RDD[T]s 
    oneAtATime: pick one rdd each time or pick all of them once.
    default:    The default rdd if no more in rdds   
    Returns DStream[T]
                                   
#From DStream to RDD 
dstream.foreachRDD(func) 
    func takes either undelying rdd or (rdd,time)
dstream.transform(func)
    func: takes either rdd[T] or timestamp,rdd[T] and returns rdd[U]
dstream.transformWith(func, other, keepSerializer=False)
    func: either rdd_a[T],rdd_b[U] or (time, rdd_a[T], rdd_b[U]) and returns rdd[V]
sparkStreamingContext.transform(dstreams, transformFunc)
    dstreams : list of DStream 
    transformFunc(list_of_rdds_from dstreams):Returns transformed list of rdds 
                                  
   
##Spark - Streaming -  UpdateStateByKey Operation
dstream.updateStateByKey(updateFunc, numPartitions=None, initialRDD=None)
            updateFunc: (list_ofV_fromRecentRDD_for_a_K, old_S) => new_S
            initialRDD[(K,V)] to seed the update process 
        Return a new 'state' ,ie RDD[(K,S)] of new DStream 
        where the state for each key,K is updated 
        by applying the updateFunc on RDD[(K,V)] of input DStream


#To use this, you will have to do two steps.
1.Define the state - The state can be an arbitrary data type.
2.Define the state update function 
  Specify with a function how to update the state 
  using the previous state and the new values from an input stream


#Example - stateful_network_wordcount_v2.py/stateful_network_wordcount.py
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext


# RDD with initial state (key, value) pairs
initialStateRDD = sc.parallelize([(u'hello', 1), (u'world', 1)])

#updateFunc – State update function. take two args , list of current value and last state 
#If this function returns None, then corresponding state key-value pair will be eliminated
def updateFunc(new_values, last_sum):
    return sum(new_values) + (last_sum or 0)

lines = ssc.socketTextStream(host, int(port))
running_counts = lines.flatMap(lambda line: line.split(" "))\
                      .map(lambda word: (word, 1))\
                      .updateStateByKey(updateFunc, initialRDD=initialStateRDD)

running_counts.pprint()

ssc.start()
ssc.awaitTermination()
    


##Spark - Streaming -  queueStream[T]- Create an input stream from a list of RDDs.
sparkStreamingContext.queueStream(rdds, oneAtATime=True, default=None)
    Create an input stream from an queue of RDDs or list. 
    In each batch, it will process either one 
    or all of the RDDs returned by the queue.
    Changes to the queue after the stream is created will not be recognized.

#Example - queue_stream.py
import time
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


rddQueue = []
for i in range(5):
    rddQueue += [ssc.sparkContext.parallelize([j for j in range(1, 1001)], 10)]

# Create the QueueInputDStream and use it do some processing
inputStream = ssc.queueStream(rddQueue)
mappedStream = inputStream.map(lambda x: (x % 10, 1))
reducedStream = mappedStream.reduceByKey(lambda a, b: a + b)
reducedStream.pprint()

ssc.start()
time.sleep(6)
ssc.stop(stopSparkContext=True, stopGraceFully=True)




##Spark - Streaming -  Transform Operation
dstream.transform(func)
        func: takes either rdd[T] or timestamp,rdd[T} and returns rdd[U]
    Return a new DStream in which each RDD[U} is generated 
    by applying a function on each RDD[T} of this DStream.
dstream.transformWith(func, other, keepSerializer=False)
        func: either rdd_a[T],rdd_b[U] or (time, rdd_a[T], rdd_b[U]) and returns rdd[V]
    Return a new DStream in which each RDD[V] is generated 
    by applying a function on each RDD[T] of this DStream and 'other' RDD[U] of DStream.



#Example - network_wordjoinsentiments.py

# Read in the word-sentiment list and create a static RDD from it
#file syntax : word     happiness_value 
word_sentiments_file_path = "data/streaming/AFINN-111.txt"
word_sentiments = ssc.sparkContext.textFile(word_sentiments_file_path) \
    .map(lambda line: tuple(line.split("\t")))

lines = ssc.socketTextStream(host, int(port))

word_counts = lines.flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b) #same Key, two values, a,b 

# Determine the words with the highest sentiment values by joining the streaming RDD
# with the static RDD inside the transform() method and then multiplying
# the frequency of the words by its sentiment value
#word_counts: (word,count) , word_sentiments= (word, happiness_value)
#after tranform, output: (word, (count,happiness_value)
happiest_words = word_counts.transform(lambda rdd: word_sentiments.join(rdd)) \ 
    .map(lambda (word, tuple): (word, float(tuple[0]) * tuple[1])) \
    .map(lambda (word, happiness): (happiness, word)) \
    .transform(lambda rdd: rdd.sortByKey(False))

happiest_words.foreachRDD(lambda rdd: print(rdd.count(), rdd.take(5)))

ssc.start()
ssc.awaitTermination()

    
    

   
    
##Spark - Streaming -  Windowing Operation    
#every time the window slides over a source DStream, 
#the source RDDs that fall within the window are combined 
#and operated upon to produce the RDDs of the windowed DStream
•window length - The duration of the window 
•sliding interval - The interval at which the window operation is performed 


# Reduce last 30 seconds of data, every 10 seconds
windowedWordCounts = pairs.reduceByKeyAndWindow(lambda x, y: x + y, lambda x, y: x - y, 30, 10)      



    
    
    
    

    
    
    
###Spark - Streaming -  DataFrame and SQL Operations on DStream 
#Can use SQL Query on DF created by sparkSession.createDataFrame(rdd)
#where rdd is DStream rdd, get RDD inside f by using  dstream.transform(f) or dstream/foreachRDD(f)



#Example - sql_network_wordcount.py
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession


#Create singleton sparkSession 
#this way you can share global singleton 
#note sharing variable is no no, use broadcast and accumulator 
def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession\
            .builder\
            .config(conf=sparkConf)\
            .getOrCreate()
    return globals()['sparkSessionSingletonInstance']



sc = SparkContext(appName="PythonSqlNetworkWordCount")
ssc = StreamingContext(sc, 1)

# Create a socket stream on target ip:port and count the
# words in input stream of \n delimited text (eg. generated by 'nc')
lines = ssc.socketTextStream(host, int(port))
words = lines.flatMap(lambda line: line.split(" "))

# Convert RDDs of the words DStream to DataFrame and run SQL query
def process(time, rdd):
    print("========= %s =========" % str(time))

    try:
        # Get the singleton instance of SparkSession
        spark = getSparkSessionInstance(rdd.context.getConf())

        # Convert RDD[String] to RDD[Row] to DataFrame
        rowRdd = rdd.map(lambda w: Row(word=w))
        wordsDataFrame = spark.createDataFrame(rowRdd)

        # Creates a temporary view using the DataFrame.
        wordsDataFrame.createOrReplaceTempView("words")

        # Do word count on table using SQL and print it
        wordCountsDataFrame = spark.sql("select word, count(*) as total from words group by word")
        wordCountsDataFrame.show()
    except:
        pass

words.foreachRDD(process)
ssc.start()
ssc.awaitTermination()


###Spark - Streaming -  Checkpointing in Stream 
#A streaming application must operate 24/7 and hence must be resilient to failures 
#For this to be possible, Spark Streaming needs to checkpoint enough information 
#to a fault- tolerant storage system such that it can recover from failures.
  
#If the checkpointDirectory exists, 
#then the context will be recreated from the checkpoint data

#Checkpointing must be enabled for below 
•Usage of stateful transformations 
        If either updateStateByKey or reduceByKeyAndWindow 
        (with inverse function) is used in the application, 
        then the checkpoint directory must be provided to allow 
        for periodic RDD checkpointing.
•Recovering from failures of the driver running the application 
    Metadata checkpoints are used to recover with progress information.

#Accumulators and Broadcast variables cannot be recovered from checkpoint in Spark Streaming
#Check http://spark.apache.org/docs/latest/streaming-programming-guide.html#accumulators-broadcast-variables-and-checkpoints


##Steps for programming with Checkpointing
#1.Function to create and setup a new StreamingContext
def functionToCreateContext():
    sc = SparkContext(...)  # new context
    ssc = StreamingContext(...)
    lines = ssc.socketTextStream(...)  # create DStreams
    ...
    ssc.checkpoint(checkpointDirectory)  # set checkpoint directory
    return ssc

#2.Get StreamingContext from checkpoint data or create a new one
#getOrCreate takes 'creation' function with no arg 
context = StreamingContext.getOrCreate(checkpointDirectory, functionToCreateContext)

# Do additional setup on context that needs to be done,
# irrespective of whether it is being started or restarted
context. ...

#3.Start the context
context.start()
context.awaitTermination()


    
#Example of checkpoint and accumulator and broadcast -recoverable_network_wordcount.py
#Accumulators and Broadcast variables cannot be recovered from checkpoint

from pyspark import SparkContext
from pyspark.streaming import StreamingContext


# Get or register a Broadcast variable
def getWordBlacklist(sparkContext):
    if ('wordBlacklist' not in globals()):
        globals()['wordBlacklist'] = sparkContext.broadcast(["a", "b", "c"])
    return globals()['wordBlacklist']


# Get or register an Accumulator
def getDroppedWordsCounter(sparkContext):
    if ('droppedWordsCounter' not in globals()):
        globals()['droppedWordsCounter'] = sparkContext.accumulator(0)
    return globals()['droppedWordsCounter']


def createContext(host, port, outputPath):
    # If you do not see this printed, that means the StreamingContext has been loaded
    # from the new checkpoint directory directly
    #Hence this function is executed for first time only 
    print("Creating new context")
    if os.path.exists(outputPath):
        os.remove(outputPath)
    sc = SparkContext(appName="PythonStreamingRecoverableNetworkWordCount")
    ssc = StreamingContext(sc, 1)

    # Create a socket stream on target ip:port and count the
    # words in input stream of \n delimited text (eg. generated by 'nc')
    lines = ssc.socketTextStream(host, port)
    words = lines.flatMap(lambda line: line.split(" "))
    wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y) #(word,count)

    #This is executed in Driver 
    def echo(time, rdd):
        # Get or register the blacklist Broadcast
        blacklist = getWordBlacklist(rdd.context)
        # Get or register the droppedWordsCounter Accumulator
        droppedWordsCounter = getDroppedWordsCounter(rdd.context)

        # Use blacklist to drop words and use droppedWordsCounter to count them
        #filterfunc is executed in worker, hence only .add()
        def filterFunc(wordCount):
            if wordCount[0] in blacklist.value:
                droppedWordsCounter.add(wordCount[1])
                False
            else:
                True

        counts = "Counts at time %s %s" % (time, rdd.filter(filterFunc).collect())
        print(counts)
        print("Dropped %d word(s) totally" % droppedWordsCounter.value)
        print("Appending to " + os.path.abspath(outputPath))
        with open(outputPath, 'a') as f:
            f.write(counts + "\n")

    wordCounts.foreachRDD(echo)
    return ssc

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: recoverable_network_wordcount.py <hostname> <port> "
              "<checkpoint-directory> <output-file>", file=sys.stderr)
        exit(-1)
    host, port, checkpoint, output = sys.argv[1:]
    ssc = StreamingContext.getOrCreate(checkpoint, lambda: createContext(host, int(port), output))
    ssc.start()
    ssc.awaitTermination()




###Spark - Streaming -  MLlib Operations on Stream (ML not available 
#Use Streaming Linear Regression, Streaming KMeans
#which can simultaneously learn from the streaming data 
#as well as apply the model on the streaming data.
    

##Spark - Streaming -  MLlib  - Streaming k-means
class pyspark.mllib.clustering.StreamingKMeansModel(clusterCenters, clusterWeights)[source]
    Clustering model which can perform an online update of the centroids
    clusterWeights
        Return the cluster weights.
    update(data, decayFactor, timeUnit)
        Update the centroids, according to data
        data – RDD with new data for the model update.
        decayFactor – Forgetfulness of the previous centroids.
        timeUnit – Can be “batches” or “points”. 
                   If points, then the decay factor is raised to the power of number 
                   of new points and if batches, then decay factor will be used as is.
    save(sc, path):
        Save this model to the given path.
    classmethod load(sc, path):
        Load a model from the given path.
    predict(x):
        Find the cluster that each of the points belongs to in this  model.
        x:
          A data point (or RDD of points) to determine cluster index.
        Returns 
          Predicted cluster index or an RDD of predicted cluster indicesif the input is an RDD.
    clusterCenters() or centers 
        Returns centers as list of points eg  (x,y) for 2D input data 
#Exmaple   
initCenters = [[0.0, 0.0], [1.0, 1.0]]
initWeights = [1.0, 1.0]
stkm = StreamingKMeansModel(initCenters, initWeights)
data = sc.parallelize([[-0.1, -0.1], [0.1, 0.1],
                       [0.9, 0.9], [1.1, 1.1]])
stkm = stkm.update(data, 1.0, u"batches")
stkm.centers
array([[ 0.,  0.],
       [ 1.,  1.]])
>>> stkm.predict([-0.1, -0.1]) #returns index of centers 
0
>>> stkm.predict([0.9, 0.9])
1
>>> stkm.clusterWeights
[3.0, 3.0]

decayFactor = 0.0
data = sc.parallelize([DenseVector([1.5, 1.5]), DenseVector([0.2, 0.2])])
stkm = stkm.update(data, 0.0, u"batches")
>>> stkm.centers
array([[ 0.2,  0.2],
       [ 1.5,  1.5]])
>>> stkm.clusterWeights
[1.0, 1.0]
>>> stkm.predict([0.2, 0.2])
0
>>> stkm.predict([1.5, 1.5])
1

pyspark.mllib.clustering.StreamingKMeans(k=2, decayFactor=1.0, timeUnit='batches')
    latestModel()
        Return the latest model
        Returns StreamingKMeansModel
    predictOn(dstream)
        Make predictions on a dstream. 
        Returns predicted(index of cluster centres) dstream 
    predictOnValues(dstream)
        Make predictions on a keyed dstream[(K,V)]. 
        Returns a predicted[(K,index of cluster centres)] dstream object.
    setDecayFactor(decayFactor)
        Set decay factor.
    setK(k)
        Set number of clusters.
    setHalfLife(halfLife, timeUnit)
        Set number of batches after which the centroids of that particular batch 
        has half the weightage.
    setInitialCenters(centers, weights)
        Set initial centers. 
        Either setInitialCenters or setRandomCenters should be called  before calling trainOn.
    setRandomCenters(dim, weight, seed)
        Set the initial centres to be random samples from a gaussian population with constant weights.
        Either setInitialCenters or setRandomCenters should be called  before calling trainOn.
    trainOn(dstream)
        Train the model on the incoming dstream.
        Keep on calling trainOn on new incoming DStream and it update model StreamingKMeansModel
        

#Example 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.clustering import StreamingKMeans


if __name__ == "__main__":
    sc = SparkContext(appName="StreamingKMeansExample")  # SparkContext
    ssc = StreamingContext(sc, 1)

    # we make an input stream of vectors for training,
    # as well as a stream of vectors for testing
    def parse(lp):
        label = float(lp[lp.find('(') + 1: lp.find(')')])
        vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))
        return LabeledPoint(label, vec)

    #syntax: 0.0 0.0 0.0
    trainingData = sc.textFile("data/mllib/kmeans_data.txt")\
        .map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))

    #syntax:(1.0), [1.7, 0.4, 0.9]
    testingData = sc.textFile("data/mllib/streaming_kmeans_data_test.txt").map(parse)

    #Convert RDD to DStream 
    trainingQueue = [trainingData]
    testingQueue = [testingData]
    trainingStream = ssc.queueStream(trainingQueue)
    testingStream = ssc.queueStream(testingQueue)

    # We create a model with random clusters and specify the number of clusters to find
    model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(3, 1.0, 0)

    # Now register the streams for training and testing and start the job,
    # printing the predicted cluster assignments on new data points as they arrive.
    model.trainOn(trainingStream)

    #predict on values of (K=label, V=vectors ) and then print DStream(K=original_level, V=prediced indexof center)
    result = model.predictOnValues(testingStream.map(lambda lp: (lp.label, lp.features)))
    result.pprint()

    ssc.start()
    ssc.stop(stopSparkContext=True, stopGraceFully=True)
    print("Final centers: " + str(model.latestModel().centers))



##Spark - Streaming -  MLlib  - StreamingLinearRegresion 
class pyspark.mllib.regression.StreamingLinearRegressionWithSGD(stepSize=0.1, numIterations=50, miniBatchFraction=1.0, convergenceTol=0.001)
    Train or predict a linear regression model on streaming data
    latestModel()
        Returns the latest LinearRegressionModel.
    predictOn(dstream)
        Returns:    DStream containing predictions. 
    predictOnValues(dstream)
        Use the model to make predictions on the values of a DStream and carry over its keys.
        Returns:    DStream containing the input keys and the predictions as values. 
    setInitialWeights(initialWeights)[source]
        Set the initial value of weights.
        This must be set before running trainOn and predictOn
    trainOn(dstream)
        Train the model on the incoming dstream.
        Keepon training on new incoming input DStream 



#Example 
#trainingDir and testDir contains files - each line format  (y,[x1,x2,x3]) 
#where y is the label and x1,x2,x3 are the features. 
#Move a file into trainingDir, Model would be updated 
#Move a file into testDir, you would see original y and predicted y 

from __future__ import print_function 
 

import sys 
from pyspark import SparkContext 
from pyspark.streaming import StreamingContext 
from pyspark.mllib.linalg import Vectors 
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD 

 
if __name__ == "__main__": 
    if len(sys.argv) != 3: 
        print("Usage: streaming_linear_regression_example.py <trainingDir> <testDir>", 
              file=sys.stderr) 
        exit(-1) 
 
    sc = SparkContext(appName="PythonLogisticRegressionWithLBFGSExample") 
    ssc = StreamingContext(sc, 1) 
 
    # $example on$ 
    def parse(lp): 
        label = float(lp[lp.find('(') + 1: lp.find(',')]) 
        vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(',')) 
        return LabeledPoint(label, vec) 
 
    trainingData = ssc.textFileStream(sys.argv[1]).map(parse).cache() 
    testData = ssc.textFileStream(sys.argv[2]).map(parse) 
 
    numFeatures = 3 
    model = StreamingLinearRegressionWithSGD() 
    model.setInitialWeights([0.0, 0.0, 0.0]) 
 
    model.trainOn(trainingData) 
    #Trasnformed (label, features) into (label, predicted_y)
    print(model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features)))) 
 
    ssc.start() 
    ssc.awaitTermination() 
 
   
   
   

    
    
    
    
/***** Advanced Streaming of Kaffka, HDFS, Flume etc  *****/
###Spark - Streaming -  Integration with HDFS 
#directory: Move a a text file to hdfs://<NameNode>:<NameNode port>/dir 
#Each line would be processed and words counts would be printed 
#Note old counts would note be cummulated, for that use updateStateByKey

sc = SparkContext(appName="PythonStreamingHDFSWordCount")
ssc = StreamingContext(sc, 60) #batchDuration should be greater than timeTaken to process files else Nothing happens 

lines = ssc.textFileStream(directory)
counts = lines.flatMap(lambda line: line.split(" "))\
              .map(lambda x: (x, 1))\
              .reduceByKey(lambda a, b: a+b)
counts.pprint()

ssc.start()
ssc.awaitTermination() 
    
#With updateStateByKey , checkpoint = is dir for storing checkpoints 
def createContext(directory, checkpoint):    
    print("Creating for first time...")
    sc = SparkContext(appName="PythonStreamingStatefulNetworkWordCount")
    ssc = StreamingContext(sc, 20)  #batch size make it big
    ssc.checkpoint(checkpoint)
    # RDD with initial state (key, value) pairs
    initialStateRDD = sc.parallelize([(u'hello', 1), (u'world', 1)])

    def updateFunc(new_values, last_sum):  #last_sum for non existing key is None
        return sum(new_values) + (last_sum or 0)

    lines = ssc.textFileStream(directory)
    running_counts = lines.flatMap(lambda line: line.split(" "))\
                          .map(lambda word: (word, 1))\
                          .updateStateByKey(updateFunc, initialRDD=initialStateRDD)

    running_counts.pprint()
    return ssc
    
    
ssc = StreamingContext.getOrCreate(checkpoint,  lambda: createContext(directory, checkpoint))
ssc.start()
ssc.awaitTermination()


###Spark - Streaming -  Integration with Kafka 
#The Kafka project introduced a new consumer api between versions 0.8 and 0.10, 
#so there are 2 separate corresponding Spark Streaming packages available
                    spark-streaming-kafka-0-8                   spark-streaming-kafka-0-10
Broker Version      0.8.2.1 or higher                           0.10.0 or higher 
Api                 stable                                      experimental 


##Reference 
class pyspark.streaming.kafka.Broker(host, port)
    Represent the host and port info for a Kafka broker.
class pyspark.streaming.kafka.KafkaMessageAndMetadata(topic, partition, offset, key, message)
    Kafka message and metadata information. Including topic, partition, offset and message
class pyspark.streaming.kafka.OffsetRange(topic, partition, fromOffset, untilOffset)
    Represents a range of offsets from a single Kafka TopicAndPartition.
class pyspark.streaming.kafka.TopicAndPartition(topic, partition)
    Represents a specific topic and partition for Kafka.
pyspark.streaming.kafka.utf8_decoder(s)
    Decode the unicode as UTF-8
                  

    
class pyspark.streaming.kafka.KafkaUtils
    classmethod createDirectStream(ssc, topics, kafkaParams, fromOffsets=None, 
                keyDecoder=<function utf8_decoder>, valueDecoder=<function utf8_decoder>, messageHandler=None)
            Returns: A DStream object    
        Create an input stream that directly pulls messages from a Kafka Broker 
        and specific offset in each batch duration and processed without storing.
        This is not a receiver based Kafka input stream, 
        This does not use Zookeeper to store offsets. 
        The consumed offsets are tracked by the stream itself. 
        For interoperability with Kafka monitoring tools that depend on Zookeeper, 
        you have to update Kafka/Zookeeper yourself from the streaming application. 
        You can access the offsets used in each batch from the generated RDDs 
        To recover from driver failures, you have to enable checkpointing in the StreamingContext. The information on consumed offset can be recovered from the checkpoint. See the programming guide for details (constraints, etc.).
        Parameters:
            •ssc – StreamingContext object.
            •topics – list of topic_name to consume.
            •kafkaParams – Additional params for Kafka.
            •fromOffsets – Per-topic/partition Kafka offsets defining the (inclusive) starting point of the stream.
            •keyDecoder – A function used to decode key (default is utf8_decoder).
            •valueDecoder – A function used to decode value (default is utf8_decoder).
            •messageHandler – A function used to convert KafkaMessageAndMetadata. You can assess meta using messageHandler (default is None).
    classmethod createRDD(sc, kafkaParams, offsetRanges, leaders=None, keyDecoder=<function utf8_decoder>, valueDecoder=<function utf8_decoder>, messageHandler=None)
        Create an RDD from Kafka using offset ranges for each topic and partition.
            •sc – SparkContext object
            •kafkaParams – Additional params for Kafka
            •offsetRanges – list of offsetRange to specify topic:partition:[start, end) to consume
            •leaders – Kafka brokers for each TopicAndPartition in offsetRanges. 
             May be an empty map, in which case leaders will be looked up on the driver.
            Returns: An RDD object     
    static createStream(ssc, zkQuorum, groupId, topics, kafkaParams=None, storageLevel=StorageLevel(True, True, False, False, 2), keyDecoder=<function utf8_decoder>, valueDecoder=<function utf8_decoder>)
        Create an input stream that pulls messages from a Kafka Broker.
        Parameters:
            •ssc – StreamingContext object
            •zkQuorum – Zookeeper quorum (hostname:port,hostname:port,..).
            •groupId – The group id for this consumer.
            •topics – Dict of (topic_name -> numPartitions) to consume. Each partition is consumed in its own thread.
            •kafkaParams – Additional params for Kafka
            •storageLevel – RDD storage level.
            •keyDecoder – A function used to decode key (default is utf8_decoder)
            •valueDecoder – A function used to decode value (default is utf8_decoder)
        Returns: A DStream object

##Spark - Kafka - Kafka and Zookeeper installations

#ZooKeeper is a distributed coordination service for distributed applications.
#Client can store data in zookeeper data node(znode) and those can be read and written atomically. 
#ZooKeeper is replicated over a sets of hosts called an ensemble

#Zookeeper provides a set of guarantees. 
#These are:
•Sequential Consistency - Updates from a client will be applied in the order that they were sent.
•Atomicity - Updates either succeed or fail. No partial results.
•Single System Image - A client will see the same view of the service regardless of the server that it connects to.
•Reliability - Once an update has been applied, it will persist from that time forward until a client overwrites the update.
•Timeliness - The clients view of the system is guaranteed to be up-to-date within a certain time bound.

#it supports below operations:
create 
    creates a node at a location in the tree
delete 
    deletes a node
exists 
    tests if a node exists at a location
get data 
    reads the data from a node
set data 
    writes data to a node
get children 
    retrieves a list of children of a node
sync 
    waits for data to be propagated

    
    
##Standalone Operation
#Download from https://archive.apache.org/dist/zookeeper/zookeeper-3.4.10/zookeeper-3.4.10.tar.gz
#And unzip and add to PATH bin dir 

#conf/zoo.cfg:
tickTime=2000
dataDir=C:/zookeeper/data
clientPort=2181

#Change the value of dataDir to specify an existing (empty to start with) directory


#start ZooKeeper:
$ bin/zkServer.cmd start

#CLI to ZooKeeper
#DataNode path is like Filesystem path 
$ bin/zkCli.cmd -server 127.0.0.1:2181
Connecting to localhost:2181
log4j:WARN No appenders could be found for logger (org.apache.zookeeper.ZooKeeper).
log4j:WARN Please initialize the log4j system properly.
Welcome to ZooKeeper!
JLine support is enabled
[zkshell: 0] help
ZooKeeper host:port cmd args
        get path [watch]
        ls path [watch]
        set path data [version]
        delquota [-n|-b] path
        quit
        printwatches on|off
        createpath data acl
        stat path [watch]
        listquota path
        history
        setAcl path acl
        getAcl path
        sync path
        redo cmdno
        addauth scheme auth
        delete path [version]
        setquota -n|-b val path

[zkshell: 8] ls /
[zookeeper]
        

#create a new znode by running create 
#This creates a new znode '/zk_test' and associates the string "my_data" with the node. 
[zkshell: 9] create /zk_test my_data
Created /zk_test
      
[zkshell: 11] ls /
[zookeeper, zk_test]

[zkshell: 12] get /zk_test
my_data
cZxid = 5
ctime = Fri Jun 05 13:57:06 PDT 2009
mZxid = 5
mtime = Fri Jun 05 13:57:06 PDT 2009
pZxid = 5
cversion = 0
dataVersion = 0
aclVersion = 0
ephemeralOwner = 0
dataLength = 7
numChildren = 0
        
#change the data associated with zk_test by issuing the set command, as in: 
[zkshell: 14] set /zk_test junk
cZxid = 5
ctime = Fri Jun 05 13:57:06 PDT 2009
mZxid = 6
mtime = Fri Jun 05 14:01:52 PDT 2009
pZxid = 5
cversion = 0
dataVersion = 1
aclVersion = 0
ephemeralOwner = 0
dataLength = 4
numChildren = 0

[zkshell: 15] get /zk_test
junk
cZxid = 5
ctime = Fri Jun 05 13:57:06 PDT 2009
mZxid = 6
mtime = Fri Jun 05 14:01:52 PDT 2009
pZxid = 5
cversion = 0
dataVersion = 1
aclVersion = 0
ephemeralOwner = 0
dataLength = 4
numChildren = 0
      
#delete the node by issuing: 
[zkshell: 16] delete /zk_test
[zkshell: 17] ls /
[zookeeper]
[zkshell: 18]



##Kafka - distributed streaming platform
1.It lets you publish and subscribe to streams of records. 
  In this respect it is similar to a message queue or enterprise messaging system. 
2.It lets you store streams of records in a fault-tolerant way. 
3.It lets you process streams of records as they occur. 

#Kafka is run as a cluster on one or more servers(called brokers)
#The Kafka cluster stores streams of records in categories called topics. 
#Each record consists of a key, a value, and a timestamp.

#Zookeeper is required for running Kafka
#it uses zookeeper for 
    Electing a controller
        The controller is one of the brokers 
        and is responsible for maintaining the leader/follower relationship 
        for all the partitions.
    Cluster membership 
        which brokers are alive and part of the cluster
    Topic configuration 
        which topics exist, how many partitions each has, where are the replicas, 
        who is the preferred leader, what configuration overrides are set for each topic
    Quotas 
        how much data is each client allowed to read and write
    ACLs 
        who is allowed to read and write to which topic (old high level consumer) 
        Which consumer groups exist, who are their members 
        and what is the latest offset each group got from each partition.
    


#At a high-level Kafka gives the following guarantees: 
• Messages sent by a producer to a particular topic partition will be appended in the order they are sent. 
  That is, if a record M1 is sent by the same producer as a record M2, 
  and M1 is sent first, then M1 will have a lower offset than M2 
  and appear earlier in the log. 
•A consumer instance sees records in the order they are stored in the log. 
•For a topic with replication factor N, kafka will tolerate up to N-1 server failures 
 without losing any records committed to the log. 


##Operations of Kafka 
#Download from https://archive.apache.org/dist/kafka/0.11.0.0/kafka_2.11-0.11.0.0.tgz
#Update PATH with <<install>>\windows\bin and C:\windows\system32\wbem


#Start the server 
$ zookeeper-server-start.bat  config/zookeeper.properties  #client binds to ..:2181
$ kafka-server-start.bat config/server.properties   #server binds to localhost:9092, called broker-list 


#Kafka can be controlled by --zookeeper quorum or by --broker-list brokers 
#Zookeeper quorum (hostname:port,hostname:port,..) of zookeeper servers 
#for standalone Zookeeper eg localhost:2181
#broker-list , list of kafka servers eg localhost:9092 

#create a topic named "test" with a single partition and only one replica
$ kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test
 
#List the topic 
$ kafka-topics.bat  --list --zookeeper localhost:2181
test
 

#Send some messages-Run the producer and then type a few messages into the console to send to the server.
$ kafka-console-producer.bat --broker-list localhost:9092 --topic test
This is a message
This is another message
 

#Start a consumer
$ kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic test --from-beginning
This is a message
This is another message
 
#Update producer terminal and check the consumer terminal 


##Kafka- Multi broker/server operations - Norte earlier broker is still running 
#https://kafka.apache.org/quickstart#quickstart_multibroker

$ cp config/server.properties config/server-1.properties
$ cp config/server.properties config/server-2.properties
 
#update - config/server-1.properties:
    broker.id=1
    listeners=PLAINTEXT://:9093
    log.dir=/tmp/kafka-logs-1
 
#update - config/server-2.properties:
    broker.id=2
    listeners=PLAINTEXT://:9094
    log.dir=/tmp/kafka-logs-2
 

#listerns: The address the socket server listens on
#FORMAT: listeners = listener_name://host_name:port
#EXAMPLE:  listeners = PLAINTEXT://your.host.name:9092

#broker.id  :  unique and permanent name of each node in the cluster. 
#by default it is zero 

#start other two nodes 
$ kafka-server-start.bat config/server-1.properties &
$ kafka-server-start.bat config/server-2.properties &

 

#create a new topic with a replication factor of three:
$ kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 3 --partitions 1 --topic my-replicated-topic
 

#run the "describe topics" command:
$ kafka-topics.bat --describe --zookeeper localhost:2181 --topic my-replicated-topic
Topic:my-replicated-topic   PartitionCount:1    ReplicationFactor:3 Configs:
    Topic: my-replicated-topic  Partition: 0    Leader: 1   Replicas: 1,2,0 Isr: 1,2,0
 

#The first line gives a summary of all the partitions, each additional line gives information about one partition. 
#Since we have only one partition for this topic there is only one line.
•"leader" is the node(broker.id) responsible for all reads and writes for the given partition. 
  Each node will be the leader for a randomly selected portion of the partitions. 
•"replicas" is the list of nodes(broker.id) that replicate the log for this partition 
  regardless of whether they are the leader or even if they are currently alive. 
•"isr" is the set of "in-sync" replicas. 
  This is the subset of the replicas list that is currently alive and caught-up to the leader. 

  
#for earlier topic 
$ kafka-topics.bat --describe --zookeeper localhost:2181 --topic test
Topic:test  PartitionCount:1    ReplicationFactor:1 Configs:
    Topic: test Partition: 0    Leader: 0   Replicas: 0 Isr: 0
 

#publish to earlier instance of broker 9092 
$ kafka-console-producer.bat --broker-list localhost:9092 --topic my-replicated-topic
...
my test message 1
my test message 2
^C
 

#consume these messages:
$ kafka-console-consumer.bat --bootstrap-server localhost:9092 --from-beginning --topic my-replicated-topic
...
my test message 1
my test message 2
^C
 

#test out fault-tolerance. Broker 1 was acting as the leader so let's kill it:
$ ps aux | grep server-1.properties
7564 ttys002    0:15.91 /System/Library/Frameworks/JavaVM.framework/Versions/1.8/Home/bin/java...
> kill -9 7564
 
#On Windows use: 
> wmic process get processid,caption,commandline | find "java.exe" | find "server-1.properties"
java.exe    java  -Xmx1G -Xms1G -server -XX:+UseG1GC ... "build\libs\kafka_2.11-0.11.0.0.jar"  kafka.Kafka config\server-1.properties    644
> taskkill /pid 644 /f
 

#check now, Leader is 2 now 
$ kafka-topics.bat --describe --zookeeper localhost:2181 --topic my-replicated-topic
Topic:my-replicated-topic   PartitionCount:1    ReplicationFactor:3 Configs:
    Topic: my-replicated-topic  Partition: 0    Leader: 2   Replicas: 1,2,0 Isr: 2,0
 

#But the messages are still available for consumption 
$ kafka-console-consumer.bat --bootstrap-server localhost:9092 --from-beginning --topic my-replicated-topic
...
my test message 1
my test message 2
^C
 
        

##Spark - Kafka - Enable Write Ahead Logs
#Check further deployment guidelines in 
#https://spark.apache.org/docs/latest/streaming-programming-guide.html#deploying-applications
1. Setting the SparkConf property spark.streaming.receiver.writeAheadLog.enable to True 
   (default is False).
   This saves data on HDFS system , hence HDFS must be enabled 
2. use KafkaUtils.createStream(..., storageLevel=StorageLevel.MEMORY_AND_DISK_SER)



##Spark - Kafka - Approach 1: Receiver-based Approach - using the Kafka high-level consumer API

#the data received from Kafka through a Receiver is stored in Spark executors, 
#and then jobs launched by Spark Streaming processes the data.

#under default configuration, this approach can lose data under failures 
#To ensure zero-data loss, enable Write Ahead Logs 
#or use Direct API 


#Example - kafka_wordcount.py <zkQuorum> <topics> 
$ spark-submit --master local[4] --jars tobeShared/py/spark-streaming-kafka-0-8-assembly_2.11-2.1.0.jar kafka_wordcount.py localhost:2181 test  

#start the producer 
#KafkaWordCountProducer <metadataBrokerList> <topic> <messagesPerSec> <wordsPerMessage>
$ java -cp "kfka-learning-assembly.jar;C:\scala\lib\*"  examples.KafkaWordCountProducer localhost:9092 test 10 2 
#or from console 
$ kafka-console-producer.bat --broker-list localhost:9092 --topic test


#Example - word count - kafka_wordcount.py
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: kafka_wordcount.py <zk> <topic>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="PythonStreamingKafkaWordCount")
    ssc = StreamingContext(sc, 1)
    ssc.checkpoint("./checkpoint")
    
    zkQuorum, topic = sys.argv[1:]
    #groupID = 'spark-streaming-consumer'
    kvs = KafkaUtils.createStream(ssc, zkQuorum, "spark-streaming-consumer", {topic: 1})
    lines = kvs.map(lambda x: x[1])
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a+b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()
    
    
    
    
##Spark - Kafka - Approach 2: Direct Approach (No Receivers)
#ensure stronger end-to-end guarantees. 
#this approach periodically queries Kafka for the latest offsets in each topic+partition, 
#and accordingly defines the offset ranges to process in each batch


#In the Kafka parameters, specify either metadata.broker.list or bootstrap.servers. 
#metadata.broker.list : The format is host1:port1,host2:port2, , list comes from kafka server.properties 

#direct_kafka_wordcount.py <broker_list> <topic>
$ spark-submit spark-submit --master local[4] --jars tobeShared/py/spark-streaming-kafka-0-8-assembly_2.11-2.1.0.jar direct_kafka_wordcount.py localhost:9092 test

#start the producer 
#KafkaWordCountProducer <metadataBrokerList> <topic> <messagesPerSec> <wordsPerMessage>
$ java -cp "kfka-learning-assembly.jar;C:\scala\lib\*" examples.KafkaWordCountProducer localhost:9092 test 10 2 
#or from console 
$ kafka-console-producer.bat --broker-list localhost:9092 --topic test


#Example - word count - direct_kafka_wordcount.py 
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: direct_kafka_wordcount.py <broker_list> <topic>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="PythonStreamingDirectKafkaWordCount")
    ssc = StreamingContext(sc, 2)

    brokers, topic = sys.argv[1:]
    kvs = KafkaUtils.createDirectStream(ssc, [topic], {"metadata.broker.list": brokers})
    lines = kvs.map(lambda x: x[1])
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a+b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()

    
    
    
    
    
    
###Spark - Streaming -  Integration with Flume
#Apache Flume is a distributed system for efficiently collecting, aggregating 
#and moving large amounts of log data from many different sources 
#to a centralized data store.

#Flume agent is a (JVM) process that hosts the components(source, channel, sink - each one could be many)
#One flow is One source to one channel to one sink 
#Source produces events based on source.type, events goes to channel before being picked by a sink.type
#A source can specify multiple channels for outputs 
#but a sink can only specify one channel for input 
#Channel - Buffer place with many inputs(each connected to a source) and one output(connected to a sink)

#Each component (source, sink or channel) has a name, type, and set of properties



#conf/first.conf.properties
#A single-node Flume configuration
# Sources, channels and sinks are defined per agent, 
# in this case named 'a1'

# Name the components on this agent eg sources, sinks, channels - names are r1,k1,c1
a1.sources = r1  
a1.sinks = k1
a1.channels = c1

# Describe/configure the source
a1.sources.r1.type = netcat
a1.sources.r1.bind = localhost
a1.sources.r1.port = 44444

# Describe the sink
a1.sinks.k1.type = logger

# Use a channel which buffers events in memory
a1.channels.c1.type = memory
a1.channels.c1.capacity = 1000
a1.channels.c1.transactionCapacity = 100

##Start the agent - named a1 
bin\flume-ng.cmd agent --conf conf --conf-file conf\first.conf.properties --name a1 -property "flume.root.logger=INFO,console"
#You can put all properties, env variable in conf\flume-env.ps1(check template conf\flume-env.ps1.template)

#Here comes logger output from all telnets commond prompts 
2018-02-16 16:55:06,085 (SinkRunner-PollingRunner-DefaultSinkProcessor) [INFO -
org.apache.flume.sink.LoggerSink.process(LoggerSink.java:95)] Event: { headers:{
} body: 68 65 6C 6C 6F 20 77 6F 72 6C 64 20 66 72 6F 6D hello world from }

##Start in two cmds - pushing data  
$ telnet localhost 44444
$ telnet localhost 44444

##Using environment variables as values in configuration files
a1.sources.r1.port = ${NC_PORT}

##Logging production environment 
-property "flume.root.logger=INFO,console;org.apache.flume.log.printconfig=true;org.apache.flume.log.rawdata=true"

##Appending more classpath 
-classpath  "value1;value2; .."

##Syntax- Configuration file syntax 
#list the sources, sinks and channels for the given agent, 
#and then point the source and sink to a channel. 


# list the sources, sinks and channels for the agent
<Agent>.sources = <Source>
<Agent>.sinks = <Sink>
<Agent>.channels = <Channel1> <Channel2>

# set channel for source
<Agent>.sources.<Source>.channels = <Channel1> <Channel2> ...
# set channel for sink
<Agent>.sinks.<Sink>.channel = <Channel1>

# properties for sources
<Agent>.sources.<Source>.<someProperty> = <someValue>
# properties for channels
<Agent>.channel.<Channel>.<someProperty> = <someValue>
# properties for sinks
<Agent>.sources.<Sink>.<someProperty> = <someValue>
 
#Example 
agent_foo.sources = avro-AppSrv-source
agent_foo.sinks = hdfs-Cluster1-sink
agent_foo.channels = mem-channel-1

# properties of avro-AppSrv-source
agent_foo.sources.avro-AppSrv-source.type = avro
agent_foo.sources.avro-AppSrv-source.bind = localhost
agent_foo.sources.avro-AppSrv-source.port = 10000

# properties of mem-channel-1
agent_foo.channels.mem-channel-1.type = memory
agent_foo.channels.mem-channel-1.capacity = 1000
agent_foo.channels.mem-channel-1.transactionCapacity = 100

# properties of hdfs-Cluster1-sink
agent_foo.sinks.hdfs-Cluster1-sink.type = hdfs
agent_foo.sinks.hdfs-Cluster1-sink.hdfs.path = hdfs://namenode/flume/webdata



##Syntax - Adding multiple flows in an agent
# list the sources, sinks and channels for the agent
<Agent>.sources = <Source1> <Source2>
<Agent>.sinks = <Sink1> <Sink2>
<Agent>.channels = <Channel1> <Channel2>

#Example 
# list the sources, sinks and channels in the agent
agent_foo.sources = avro-AppSrv-source1 exec-tail-source2
agent_foo.sinks = hdfs-Cluster1-sink1 avro-forward-sink2
agent_foo.channels = mem-channel-1 file-channel-2

# flow #1 configuration
agent_foo.sources.avro-AppSrv-source1.channels = mem-channel-1
agent_foo.sinks.hdfs-Cluster1-sink1.channel = mem-channel-1

# flow #2 configuration
agent_foo.sources.exec-tail-source2.channels = file-channel-2
agent_foo.sinks.avro-forward-sink2.channel = file-channel-2



##Syntax - Fan out syntax 
# List the sources, sinks and channels for the agent
<Agent>.sources = <Source1>
<Agent>.sinks = <Sink1> <Sink2>
<Agent>.channels = <Channel1> <Channel2>

# set list of channels for source (separated by space)
<Agent>.sources.<Source1>.channels = <Channel1> <Channel2>

# set channel for sinks
<Agent>.sinks.<Sink1>.channel = <Channel1>
<Agent>.sinks.<Sink2>.channel = <Channel2>

<Agent>.sources.<Source1>.selector.type = replicating


##Flume - sources 
#https://flume.apache.org/FlumeUserGuide.html#flume-sources
#type           Required properties 
avro            bind, port,channels 
thrift          bind,port,channels
exec            command,channels  #executing external command eg command=tail -F /var/log/secure
jms             channels,initialContextFactory ,connectionFactory ,providerURL ,destinationName ,destinationType 
spooldir        channels,spoolDir #The directory from which to read files from. 
TAILDIR         channels,filegroups, filegroups.<filegroupName> #put absolute REGEX file path in filegroups.<filegroupName>
netcat          channels,bind, port  #TCP Source
netcatudp       channels,bind, port  #UDP Source
org.apache.flume.source.kafka.KafkaSource   channels,kafka.bootstrap.servers,kafka.topics,kafka.topics.regex
seq             channels      #A simple sequence generator that continuously generates events with a counter that starts from 0, increments by 1 and stops at totalEvents
syslogtcp       channels,host, port  #Syslog TCP Source
syslogudp       channels,host, port  #Syslog UDP Source
http            port        #binds to all ips on this host and handles JSON(array of events) by default 

##Flume - Sinks
#https://flume.apache.org/FlumeUserGuide.html#flume-sinks
hdfs            channel,hdfs.path  #eg hdfs://namenode/flume/webdata/, supports many formatting chars 
hive            channel,hive.metastore,hive.database,hive.table 
logger          channel        #Logs event at INFO level
avro            channel,hostname,port
thrift          channel,hostname,port
irc             channel,hostname,noc,chan  #IRC sink
file_roll       channel,sink.directory  #The directory where files will be stored
null            channel       #null sink 
hbase           channel,table, columnFamily
org.apache.flume.sink.kafka.KafkaSink       kafka.bootstrap.servers,kafka.topic #This is a Flume Sink implementation that can publish data to a Kafka topic
http            channel,endpoint #The fully qualified URL endpoint to POST to

##Flume - Channels
#https://flume.apache.org/FlumeUserGuide.html#flume-channels
memory          capacity  #The events are stored in an in-memory queue with configurable max size
jdbc            db.username.db.password  #Supports only DERBY
file                    #File Channel
org.apache.flume.channel.kafka.KafkaChannel         kafka.bootstrap.servers


##Reference 
class pyspark.streaming.flume.FlumeUtils
    classmethod createPollingStream(ssc, addresses, storageLevel=StorageLevel(True, True, False, False, 2), 
                maxBatchSize=1000, parallelism=5, bodyDecoder=<function utf8_decoder at 0x7f51eb4416e0>)
        Creates an input stream that is to be used with the Spark Sink deployed on a Flume agent. 
        This stream will poll the sink for data and will pull events as they are available.
        Parameters:
            •ssc – StreamingContext object
            •addresses – List of (host, port)s on which the Spark Sink is running.
            •storageLevel – Storage level to use for storing the received objects
            •maxBatchSize – The maximum number of events to be pulled from the Spark sink in a single RPC call
            •parallelism – Number of concurrent requests this stream should send to the sink. Note that having a higher number of requests concurrently being pulled will result in this stream using more threads
            •bodyDecoder – A function used to decode body (default is utf8_decoder)
        Returns:    A DStream object 
    classmethod createStream(ssc, hostname, port, storageLevel=StorageLevel(True, True, False, False, 2), 
                enableDecompression=False, bodyDecoder=<function utf8_decoder at 0x7f51eb4416e0>)
        Create an input stream that pulls events from Flume.
        Parameters:
        •ssc – StreamingContext object
        •hostname – Hostname of the slave machine to which the flume data will be sent
        •port – Port of the slave machine to which the flume data will be sent
        •storageLevel – Storage level to use for storing the received objects
        •enableDecompression – Should netty server decompress input stream
        •bodyDecoder – A function used to decode body (default is utf8_decoder)
        Returns:    A DStream object
        
        
        
##Spark - Flume - Approach 1: Flume-style Push-based Approach        
#Flume is designed to push data between Flume agents. 
#In this approach, Spark Streaming essentially sets up a receiver that acts an Avro agent for Flume, to which Flume can push the data.
       
#Choose a machine in your cluster such that
•When your Flume + Spark Streaming application is launched, 
 one of the Spark workers must run on that machine.
•Flume can be configured to push data to a port on that machine.

#Due to the push model, the streaming application needs to be up, 
#with the receiver scheduled and listening on the chosen port, for Flume to be able push data.

##Configuration file - Fanout configuration 
#Note hat the hostname should be the same as the one used 
#by the resource manager in the cluster (Mesos, YARN or Spark Standalone), 
agent.sources = r1  
agent.sinks = avroSink k1
agent.channels = memoryChannel c1

# Describe/configure the source
agent.sources.r1.type = netcat
agent.sources.r1.bind = localhost
agent.sources.r1.port = 44444
agent.sources.r1.channels = memoryChannel c1
agent.sources.r1.selector.type = replicating

# Describe the sink
agent.sinks.k1.type = logger
agent.sinks.k1.channel = c1

agent.sinks.avroSink.type = avro
agent.sinks.avroSink.channel = memoryChannel
agent.sinks.avroSink.hostname = localhost 
agent.sinks.avroSink.port = 12345
     
agent.channels.memoryChannel.type = memory
agent.channels.memoryChannel.capacity = 1000
agent.channels.memoryChannel.transactionCapacity = 100       


# Use a channel which buffers events in memory
agent.channels.c1.type = memory
agent.channels.c1.capacity = 1000
agent.channels.c1.transactionCapacity = 100     
        
        
        
##Execution         
$ spark-submit --jars tobeShared/py/spark-streaming-flume-assembly_2.11-2.1.0.jar flume_wordcount.py localhost 12345    
#start flume 
$ bin\flume-ng.cmd agent --conf conf --conf-file conf\spark.conf.properties --name agent 
#Start source 
$ telnet localhost 44444



"""
 Counts words in UTF8 encoded, '\n' delimited text received from the network every second.
"""

from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.flume import FlumeUtils

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: flume_wordcount.py <hostname> <port>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="PythonStreamingFlumeWordCount")
    ssc = StreamingContext(sc, 1)

    hostname, port = sys.argv[1:]
    kvs = FlumeUtils.createStream(ssc, hostname, int(port))
    lines = kvs.map(lambda x: x[1])
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a+b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()   
    
    
    
##Spark - Flume - Approach 2:   using a Custom Sink  
#uses spark specific custom sink 

#Configuration 
1. download spark-streaming-flume-sink_2.11-2.2.1.jar
2. download commons-lang3-3.5.jar
3. download scala-library-2.11.8.jar
4. put them in C:\flume\lib (commandline options --classpath does not work)
   Delete old scala-library-2xx version 


#Configuration file : spark2.conf.properties 
agent.sources = r1  
agent.sinks = spark k1
agent.channels = memoryChannel c1

# Describe/configure the source
agent.sources.r1.type = netcat
agent.sources.r1.bind = localhost
agent.sources.r1.port = 44444
agent.sources.r1.channels = memoryChannel c1
agent.sources.r1.selector.type = replicating

# Describe the sink
agent.sinks.k1.type = logger
agent.sinks.k1.channel = c1

agent.sinks.spark.type = org.apache.spark.streaming.flume.sink.SparkSink
agent.sinks.spark.channel = memoryChannel
agent.sinks.spark.hostname = localhost 
agent.sinks.spark.port = 12345
     
agent.channels.memoryChannel.type = memory
agent.channels.memoryChannel.capacity = 1000
agent.channels.memoryChannel.transactionCapacity = 100       
# Use a channel which buffers events in memory
agent.channels.c1.type = memory
agent.channels.c1.capacity = 1000
agent.channels.c1.transactionCapacity = 100  


##Code change Only update below 
#Multiple agent sinks hostname and port can be given 
kvs = FlumeUtils.createPollingStream(ssc, [(hostname, int(port)),] )

    
    
    
    
   
###Spark - Streaming -  Integration with Kinesis 
#Amazon Kinesis is a fully managed service for real-time processing of streaming data at massive scale. 
#The Kinesis receiver creates an input DStream using the Kinesis Client Library (KCL) provided by Amazon under the Amazon Software License (ASL).

#https://docs.aws.amazon.com/streams/latest/dev/before-you-begin.html
#AWS CLI 
#https://docs.aws.amazon.com/streams/latest/dev/kinesis-tutorial-cli-installation.html

$ pip install awscli 
$ aws kinesis help
$ aws configure
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-west-2
Default output format [None]: json

#Step 1: Create a Stream
$ aws kinesis create-stream --stream-name Foo --shard-count 1
$ aws kinesis describe-stream --stream-name Foo
$ aws kinesis list-streams

#Step 2: Put a Record
a$ ws kinesis put-record --stream-name Foo --partition-key 123 --data testdata

#Step 3: Get the Record
#Before you can get data from the stream ,obtain the shard iterator 
#for the shard you are interested in. 
#A shard iterator represents the position of the stream and shard from which the consumer (get-record command in this case) will read. 
$ aws kinesis get-shard-iterator --shard-id shardId-000000000000 --shard-iterator-type TRIM_HORIZON --stream-name Foo
$ aws kinesis get-records --shard-iterator AAAAAAAAAAHSywljv0zEgPX4NyKdZ5wryMzP9yALs8NeKbUjp1IxtZs1Sp+KEd9I6AJ9ZG4lNR1EMi+9Md/nHvtLyxpfhEzYvkTZ4D9DQVz/mBYWRO6OTZRKnW9gd+efGN2aHFdkH1rJl4BL9Wyrk+ghYG22D2T1Da2EyNSH1+LAbK33gQweTJADBdyMwlo5r6PqcP2dzhg=

#Step 4: Clean Up
$ aws kinesis delete-stream --stream-name Foo
    
    
##Reference   
    
class pyspark.streaming.kinesis.KinesisUtils
    classmethod  createStream(ssc, kinesisAppName, streamName, endpointUrl, regionName, 
                initialPositionInStream, checkpointInterval, storageLevel=StorageLevel(True, True, False, False, 2), 
                awsAccessKeyId=None, awsSecretKey=None, decoder=<function utf8_decoder at 0x7f51e9541d70>, stsAssumeRoleArn=None, stsSessionName=None, stsExternalId=None)
    Create an input stream that pulls messages from a Kinesis stream. 
    This uses the Kinesis Client Library (KCL) to pull messages from Kinesis.
    The given AWS credentials will get saved in DStream checkpoints 
    if checkpointing is enabled. 
    Make sure that your checkpoint directory is secure.
    Parameters:
        •ssc – StreamingContext object
        •initialPositionInStream – In the absence of Kinesis checkpoint info, 
                    this is the worker’s initial starting position in the stream. 
                    The values are either the beginning of the stream per Kinesis’ limit
                    of 24 hours (InitialPositionInStream.TRIM_HORIZON) 
                    or the tip of the stream (InitialPositionInStream.LATEST).
        •storageLevel – Storage level to use for storing the received objects (default is StorageLevel.MEMORY_AND_DISK_2)
        •awsAccessKeyId – AWS AccessKeyId (default is None). 
        •awsSecretKey – AWS SecretKey (default is None). 
                Uses DefaultAWSCredentialsProviderChain to find credentials  in the following order:
                  Environment Variables - AWS_ACCESS_KEY_ID and AWS_SECRET_KEY
                  Java System Properties - aws.accessKeyId and aws.secretKey
                  Credential profiles file - default location (~/.aws/credentials) shared by all AWS SDKs
                  Instance profile credentials - delivered through the Amazon EC2 metadata service
        [Kinesis app name], kinesisAppName
            The application name that will be used to checkpoint the Kinesis sequence numbers in DynamoDB table(AWS). 
                1.The application name must be unique for a given account and region.
                2.If the table exists but has incorrect checkpoint information (for a different stream, or old expired sequenced numbers), 
                  then there may be temporary errors.
        [Kinesis stream name],streamName
            The Kinesis stream that this streaming application will pull data from.
        [endpoint URL],endpointUrl
            Valid Kinesis endpoints URL can be found here.
            http://docs.aws.amazon.com/general/latest/gr/rande.html#ak_region
        [region name],regionName
            Valid Kinesis region names can be found here.
            https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html
        [checkpoint interval],checkpointInterval
            The interval (e.g., Duration(2000) = 2 seconds) at which the Kinesis Client Library 
            saves its position in the stream. 
            For starters, set it to the same as the batch interval of the streaming application.
        •decoder – A function used to decode value (default is utf8_decoder)
        •stsAssumeRoleArn – ARN of IAM role to assume when using STS sessions to read from the Kinesis stream (default is None).
        •stsSessionName – Name to uniquely identify STS sessions used to read from Kinesis stream, if STS is being used (default is None).
        •stsExternalId – External ID that can be used to validate against the assumed IAM role’s trust policy, if STS is being used (default is None).
    Returns:A DStream object    
    
class pyspark.streaming.kinesis.InitialPositionInStream[source]
    LATEST = 0
    TRIM_HORIZON = 1   
    
    
##Details of the arguments 

##Example:
#export AWS keys if necessary
$ set AWS_ACCESS_KEY_ID=<your-access-key>
$ set AWS_SECRET_KEY=<your-secret-key>

# run the example
$ bin/spark-submit --jars tobeShared/py/spark-streaming-kinesis-asl-assembly_2.11-2.0.0.jar kinesis_wordcount_asl.py myAppName mySparkStream https://kinesis.ap-south-1.amazonaws.com ap-south-1

#To generate random string data to put onto the Kinesis stream, in another terminal, 
$ bin/run-example streaming.KinesisWordProducerASL mySparkStream https://kinesis.ap-south-1.amazonaws.com 1000 10

   
    
    
##Code example 
"""
  Consumes messages from a Amazon Kinesis streams and does wordcount.

  This code """
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kinesis import KinesisUtils, InitialPositionInStream

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: kinesis_wordcount_asl.py <app-name> <stream-name> <endpoint-url> <region-name>",
            file=sys.stderr)
        sys.exit(-1)

    sc = SparkContext(appName="PythonStreamingKinesisWordCountAsl")
    ssc = StreamingContext(sc, 1)
    appName, streamName, endpointUrl, regionName = sys.argv[1:]
    lines = KinesisUtils.createStream(ssc, appName, streamName, endpointUrl, regionName, InitialPositionInStream.LATEST, 2)
    counts = lines.flatMap(lambda line: line.split(" ")) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a+b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()
    
    
    
    
    
    

    
###Spark - Streaming - Structured Stream - based on sql.DataFrame 
#https://spark.apache.org/docs/2.2.0/structured-streaming-programming-guide.html

#stream processing engine built on the Spark SQL engine
#API of DataFrameReader/Writer(non-streaming version) are similar 

#Note 'path' in all methods can be directory, 
#then any file moved to the directory would be returned as DF continuousely 

#Here DataFrame represents an unbounded table containing the streaming data
#which is continuousely updated as and when data arrives 

#For example - For format=text, DataFrame is unbounded table of text data
#This table contains one column of strings named "value", 
#and each line in the streaming text data becomes a row in the table


##Spark - Streaming - Structured Stream - General Steps 
1. Use spark.readStream.xyz(..) or spark.readStream.format(..).option(k,v).load()
   to read continuousely from a file stream 
   (eg file getting continuousely updated or a directory containing files getting 'moved')
2. Use DataFrame methods to process that 
   Most of the common operations on DataFrame are supported for streaming
   Also windows operations are supported eg 
    windowedCounts = words.groupBy(
                window(words.timestamp, "10 minutes", "5 minutes"),
                words.word
            ).count()
3. use DataFrame.writeStream.format(..).option(k,v).trigger(..).queryName(..).outputMode(..).start()
   to start the stream 
4. start() returns  StreamingQuery, which can be used to stop()/awaitTermination() etc 
5. If queryName is mentioned, can do sql query operations as well 
   by spark.sql("select ... from queryName")



##Spark - Streaming - Structured Stream - Handling Late Data and Watermarking
#if one of the events arrives late to the application.
#For example, say, a word generated at 12:04 (i.e. event time) could be received 
#by the application at 12:11. 
#The application should use the time 12:04 instead of 12:11 to update the older counts for the window 12:00 - 12:10.

#watermarking :which lets the engine automatically track the current event time 
#in the data and attempt to clean up old state accordingly

##Conditions 
1.If DF does not contain any time columns
  Include "includeTimestamp" option , the DF contains one column as "timestamp"
2.The aggregation must have either the event-time column, 
  or a window on the event-time column
3.Output mode must be Append or Update
4.withWatermark must be called on the same column as the timestamp column 
  used in the aggregate. 
  For example, df.withWatermark("time", "1 min").groupBy("time2").count() is invalid in Append output mode, as watermark is defined on a different column from the aggregation column.
5.withWatermark must be called before the aggregation 
  for the watermark details to be used. 
  For example, df.groupBy("time").count().withWatermark("time", "1 min") is invalid in Append output mode.

#Example 
words = ...  # streaming DataFrame of schema { timestamp: Timestamp, word: String }

# Group the data by window and word and compute the count of each group
windowedCounts = words \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(words.timestamp, "10 minutes", "5 minutes"),
        words.word) \
    .count()

    
    
##Spark - Streaming - Structured Stream - Join Operations
#Streaming DataFrames can be joined with static DataFrames to create new streaming DataFrames. Here are a few examples.

staticDf = spark.read. ...
streamingDf = spark.readStream. ...
streamingDf.join(staticDf, "type")  # inner equi-join with a static DF
streamingDf.join(staticDf, "type", "right_join")  # right outer join with a static DF
  
  
  
##Spark - Streaming - Structured Stream - Streaming Deduplication
#You can deduplicate records in data streams using a unique identifier in the events
#you can use deduplication with or without watermarking

#Example 
streamingDf = spark.readStream. ...

#Without watermark using guid column
streamingDf.dropDuplicates("guid")

#With watermark using guid and eventTime columns
streamingDf \
  .withWatermark("eventTime", "10 seconds") \
  .dropDuplicates("guid", "eventTime")
  
  
  
##Spark - Streaming - Structured Stream - Not supported DF operations 
•Multiple streaming aggregations (i.e. a chain of aggregations on a streaming DF) are not yet supported on streaming Datasets.
•Limit and take first N rows are not supported on streaming Datasets.
•Distinct operations on streaming Datasets are not supported.
•Sorting operations are supported on streaming Datasets only after an aggregation and in Complete Output Mode.
•Outer joins between a streaming and a static Datasets are conditionally supported.
    ◦Full outer join with a streaming Dataset is not supported
    ◦Left outer join with a streaming Dataset on the right is not supported
    ◦Right outer join with a streaming Dataset on the left is not supported
•Any kind of joins between two streaming Datasets is not yet supported.
•count() - Cannot return a single count from a streaming Dataset. Instead, use ds.groupBy().count() which returns a streaming Dataset containing a running count.
•foreach() - Instead use ds.writeStream.foreach(...) (only for Scala and Java)
•show() - Instead use the console sink   
  
   
    
##Spark - Streaming - Structured Stream -  Output Sinks

#File sink - Stores the output to a directory.
writeStream
    .format("parquet")        # can be "orc", "json", "csv", etc.
    .option("path", "path/to/destination/dir")
    .start()
    
#Foreach sink - Runs arbitrary computation on the records in the output.
#only for Scala and Java
#implement the interface ForeachWriter , 
#which has methods that get called whenever there is a sequence of rows generated as output after a trigger
#check - https://spark.apache.org/docs/2.2.0/structured-streaming-programming-guide.html#using-foreach
writeStream
    .foreach(...)
    .start()
    
#Console sink (for debugging) - Prints the output to the console/stdout every time there is a trigger. Both, Append and Complete output modes, are supported. This should be used for debugging purposes on low data volumes as the entire output is collected and stored in the driver’s memory after every trigger.
writeStream
    .format("console")
    .start()
    
#Memory sink (for debugging) - The output is stored in memory as an in-memory table. 
#Both, Append and Complete output modes, are supported.
writeStream
    .format("memory")
    .queryName("tableName")
    .start()

    
    
##Spark - Streaming - Structured Stream - Suported OutputMode 
Queries with aggregation 
    Aggregation on event-time with watermark 
        Append, Update, Complete 
    Other aggregations 
        Complete, Update 
Other queries 
    Append, Update        
File Sink 
    Append 
    Options : 
        path: path to the output directory, must be specified     
Foreach Sink 
    Append, Update, Compelete     
Console Sink 
    Append, Update, Complete 
    Options:
        numRows: Number of rows to print every trigger (default: 20) 
        truncate: Whether to truncate the output if too long (default: true)  No  
Memory Sink 
    Append, Complete 
    
    
# ========== DF with no aggregations ==========
noAggDF = deviceDataDf.select("device").where("signal > 10")   

# Print new data to console
noAggDF \
    .writeStream \
    .format("console") \
    .start()

# Write new data to Parquet files
noAggDF \
    .writeStream \
    .format("parquet") \
    .option("checkpointLocation", "path/to/checkpoint/dir") \
    .option("path", "path/to/destination/dir") \
    .start()

# ========== DF with aggregation ==========
aggDF = df.groupBy("device").count()

# Print updated aggregations to console
aggDF \
    .writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

# Have all the aggregates in an in memory table. The query name will be the table name
aggDF \
    .writeStream \
    .queryName("aggregates") \
    .outputMode("complete") \
    .format("memory") \
    .start()

spark.sql("select * from aggregates").show()   # interactively query in-memory table   
    
    
##Spark - Streaming - Structured Stream - Recovering from Failures with Checkpointing  
#Example   
aggDF \
    .writeStream \
    .outputMode("complete") \
    .option("checkpointLocation", "path/to/HDFS/dir") \
    .format("memory") \
    .start() 
    
    
    
    
##Spark - Streaming - Structured Stream - Reference 
sdf = spark.readStream.format('text').load('python/test_support/sql/streaming')
sdf_schema = StructType([StructField("data", StringType(), False)])
df = spark.readStream.format('text').load('python/test_support/sql/streaming')


class pyspark.sql.streaming.DataStreamReader(spark)
        Use spark.readStream to access this.
        Note spark.read returns DataFrameReader
        All these methods return sql.DataFrame 
    csv(path, schema=None, sep=None, encoding=None, quote=None, 
            escape=None, comment=None, header=None, inferSchema=None, 
            ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, 
            nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, 
            dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, 
            maxMalformedLogPerPartition=None, mode=None)
        Loads a CSV file stream and returns the result as a DataFrame.
    format(source)
        Specifies the input data source format.
        source – string, name of the data source, e.g. 'socket', 'json', 'parquet','text','csv'
    json(path, schema=None, primitivesAsString=None, prefersDecimal=None, 
            allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, 
            allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, 
            mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None)
        Loads a JSON file stream (JSON Lines text format or newline-delimited JSON) 
        and returns a :class`DataFrame`.
    load(path=None, format=None, schema=None, **options)
        Loads a data stream from a data source and returns it as a :class`DataFrame`.
        >>> json_sdf = spark.readStream.format("json") \
                .schema(sdf_schema) \
                .load(tempfile.mkdtemp())
        >>> json_sdf.isStreaming
        True
        >>> json_sdf.schema == sdf_schema
        True
    option(key, value)
        Adds an input option for the underlying data source.
        >>> s = spark.readStream.option("x", 1)
    options(**options)
        Adds input options for the underlying data source.
        >>> s = spark.readStream.options(x="1", y=2)
    parquet(path)
        Loads a Parquet file stream, returning the result as a DataFrame.
        >>> parquet_sdf = spark.readStream.schema(sdf_schema).parquet(tempfile.mkdtemp())
        >>> parquet_sdf.isStreaming
        True
        >>> parquet_sdf.schema == sdf_schema
        True
    schema(schema)
        Specifies the input schema.
    text(path)
        Loads a text file stream and returns a DataFrame 
        whose schema starts with a string column named 'value', 
        and followed by partitioned columns if there are any.
        Each line in the text file is a new row in the resulting DataFrame.
        >>> text_sdf = spark.readStream.text(tempfile.mkdtemp())
        >>> text_sdf.isStreaming
        True
        >>> "value" in str(text_sdf.schema)
        True



class pyspark.sql.streaming.DataStreamWriter(df)
        Use DataFrame.writeStream to access this.
    format(source)
        Specifies the underlying output data source.
        source – string, name of the data source, which for now can be 'parquet', 'console'
        >>> writer = sdf.writeStream.format('json')
    option(key, value)
        Adds an output option for the underlying data source.
    options(**options)
        Adds output options for the underlying data source.
    outputMode(outputMode)
        Specifies how data of a streaming DataFrame/Dataset is written to a streaming sink.
        Options include:
            append:Only the new rows in the streaming DataFrame/Dataset will be written tothe sink
            complete:All the rows in the streaming DataFrame/Dataset will be written to the sinkevery time these is some updates
            update:only the rows that were updated in the streaming DataFrame/Dataset will bewritten to the sink every time there are some updates. If the query doesn’t contain aggregations, it will be equivalent to append mode.
    partitionBy(*cols)
        Partitions the output by the given columns on the file system.
    queryName(queryName)
        Specifies the name of the StreamingQuery that can be started with start(). 
        This name must be unique among all the currently active queries in the associated SparkSession.
        >>> writer = sdf.writeStream.queryName('streaming_query')
    start(path=None, format=None, partitionBy=None, queryName=None, **options)
        Streams the contents of the DataFrame to a data source.
        •format – the format used to save           
        •partitionBy – names of partitioning columns
        •queryName – unique name for the query
        •options – All other string options. You may want to provide a checkpointLocation for most streams, however it is not required for a memory stream.
        Returns StreamingQuery
        >>> sq = sdf.writeStream.format('memory').queryName('this_query').start()
        >>> sq.isActive
        True
        >>> sq.name
        u'this_query'
        >>> sq.stop()
        >>> sq.isActive
        False
        >>> sq = sdf.writeStream.trigger(processingTime='5 seconds').start(
            queryName='that_query', format='memory')
        >>> sq.name
        u'that_query'
        >>> sq.isActive
        True
        >>> sq.stop()
    trigger(*args, **kwargs)
        Set the trigger for the stream query. 
        If this is not set it will run the query as fast as possible, 
        which is equivalent to setting the trigger to processingTime='0 seconds'.
        processingTime 
            a processing time interval as a string, e.g. '5 seconds', '1 minute'. 
        >>> # trigger the query for execution every 5 seconds
        >>> writer = sdf.writeStream.trigger(processingTime='5 seconds')
        
        

class pyspark.sql.streaming.StreamingQuery(jsq)
    A handle to a query that is executing continuously in the background as new data arrives.
    Returned from DF.writeStream.start()
    awaitTermination(timeout=None)
        Waits for the termination of this query, either by query.stop() or by an exception
    exception()
        Returns:the StreamingQueryException if the query was terminated by an exception, or None. 
    explain(extended=False)
        Prints the (logical and physical) plans to the console for debugging purpose.
        Parameters:
            extended – boolean, default False. If False, prints only the physical plan. 
        >>> sq = sdf.writeStream.format('memory').queryName('query_explain').start()
        >>> sq.processAllAvailable() # Wait a bit to generate the runtime plans.
        >>> sq.explain()
        == Physical Plan ==
        ...
        >>> sq.explain(True)
        == Parsed Logical Plan ==
        ...
        == Analyzed Logical Plan ==
        ...
        == Optimized Logical Plan ==
        ...
        == Physical Plan ==
        ...
        >>> sq.stop()
    id
        Returns the unique id of this query that persists across restarts from checkpoint data. That is, this id is generated when a query is started for the first time, and will be the same every time it is restarted from checkpoint data. There can only be one query with the same id active in a Spark cluster. Also see, runId.
    isActive
        Whether this streaming query is currently active or not.
    lastProgress
        Returns the most recent StreamingQueryProgress update of this streaming query or None if there were no progress updates 
        returns a map
        query = ...  # a StreamingQuery
        >>> print(query.lastProgress)
        {u'stateOperators': [], u'eventTime': {u'watermark': u'2016-12-14T18:45:24.873Z'}, u'name': u'MyQuery', u'timestamp': u'2016-12-14T18:45:24.873Z', u'processedRowsPerSecond': 200.0, u'inputRowsPerSecond': 120.0, u'numInputRows': 10, u'sources': [{u'description': u'KafkaSource[Subscribe[topic-0]]', u'endOffset': {u'topic-0': {u'1': 134, u'0': 534, u'3': 21, u'2': 0, u'4': 115}}, u'processedRowsPerSecond': 200.0, u'inputRowsPerSecond': 120.0, u'numInputRows': 10, u'startOffset': {u'topic-0': {u'1': 1, u'0': 1, u'3': 1, u'2': 0, u'4': 1}}}], u'durationMs': {u'getOffset': 2, u'triggerExecution': 3}, u'runId': u'88e2ff94-ede0-45a8-b687-6316fbef529a', u'id': u'ce011fdc-8762-4dcb-84eb-a77333e28109', u'sink': {u'description': u'MemorySink'}}
    name
        Returns the user-specified name of the query, or null if not specified. 
        This name can be specified in the org.apache.spark.sql.streaming.DataStreamWriter 
        as dataframe.writeStream.queryName("query").start(). 
        This name, if set, must be unique across all active queries.
    processAllAvailable()
        Blocks until all available data in the source has been processed 
        and committed to the sink. This method is intended for testing.
        In the case of continually arriving data, this method may block forever. 
    recentProgress
        Returns an array of the most recent StreamingQueryProgress updates for this query. 
        The number of progress updates retained for each stream is configured 
        by Spark session configuration spark.sql.streaming.numRecentProgressUpdates.
    runId
        Returns the unique id of this query that does not persist across restarts.
        That is, every query that is started (or restarted from checkpoint) 
        will have a different runId.
    status
        Returns the current status of the query.
        >>> print(query.status)
            {u'message': u'Waiting for data to arrive', u'isTriggerActive': False, u'isDataAvailable': False}
    stop()
        Stop this streaming query.

    
    
    
class pyspark.sql.streaming.StreamingQueryManager(jsqm)
    A class to manage all the StreamingQuery StreamingQueries active.
    Returned from spark.streams 
    active
        Returns a list of active queries associated with this SQLContext
        >>> sq = sdf.writeStream.format('memory').queryName('this_query').start()
        >>> sqm = spark.streams
        >>> # get the list of active streaming queries
        >>> [q.name for q in sqm.active]
        [u'this_query']
        >>> sq.stop()
    awaitAnyTermination(timeout=None)
        Wait until any of the queries on the associated SQLContext has terminated 
        since the creation of the context, or since resetTerminated() was called. 
    get(id)
        Returns an active query from this SQLContext or throws exception if an active query with this name doesn’t exist.
        >>> sq = sdf.writeStream.format('memory').queryName('this_query').start()
        >>> sq.name
        u'this_query'
        >>> sq = spark.streams.get(sq.id)
        >>> sq.isActive
        True
        >>> sq = sqlContext.streams.get(sq.id)
        >>> sq.isActive
        True
        >>> sq.stop()
    resetTerminated()
        Forget about past terminated queries 
        so that awaitAnyTermination() can be used again to wait for new terminations.
        >>> spark.streams.resetTerminated()


##Spark - Streaming - Structured Stream - File source 
Reads files written in a directory as a stream of data
Supports glob paths, but does not support multiple comma-separated paths/globs. 
#Options 
    path: path to the input directory, and common to all file formats. 
    maxFilesPerTrigger: maximum number of new files to be considered in every trigger (default: no max) 
    latestFirst:    whether to processs the latest new files first, (default: false) 
    fileNameOnly:   whether to check new files based on only the filename instead of on the full path (default: false). 
                    With this set to `true`, the following files would be considered as the same file, 
                    because their filenames, "dataset.txt", are the same: 
                     · "file:///dataset.txt"
                     · "s3://a/dataset.txt"
                     · "s3n://a/b/dataset.txt"
                     · "s3a://a/b/c/dataset.txt"
        

##Spark - Streaming - Structured Stream - format name - socket 
#DataFrame two fields - value and timestamp if "includeTimestamp" option is set to true 
#socket format has  two options  - host and port 


##basic example  - network_wordcount_using_StructuredStream.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split

# Create DataFrame representing the stream of input lines from connection to host:port
lines = spark\
    .readStream\
    .format('socket')\
    .option('host', host)\
    .option('port', port)\
    .load()

#For example to get each word as row 
#use split and explode, to split each line into multiple words and then to rows with each word 

words = lines.select(
    # explode turns each item in an array into a separate row
    explode(
        split(lines.value, ' ')
    ).alias('word')
)

# Generate running word count
wordCounts = words.groupBy('word').count()

# Start running the query that prints the running counts to the console
query = wordCounts\
    .writeStream\
    .outputMode('complete')\
    .format('console')\
    .start()

query.awaitTermination()


   
##Example with window -network_wordcount_windowed_using_StructuredStream

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import window


# Create DataFrame representing the stream of input lines from connection to host:port
lines = spark\
    .readStream\
    .format('socket')\
    .option('host', host)\
    .option('port', port)\
    .option('includeTimestamp', 'true')\
    .load()

# Split the lines into words, retaining timestamps
# split() splits each line into an array, and explode() turns the array into multiple rows
words = lines.select(
    explode(split(lines.value, ' ')).alias('word'),
    lines.timestamp
)

# Group the data by window and word and compute the count of each group
#The output column will be a struct called 'window' by default 
#with the nested columns 'start' and 'end', where 'start' and 'end' will be of sql.types.TimestampType.

windowedCounts = words.groupBy(
    window(words.timestamp, windowDuration, slideDuration), #words.timestamp is converted to two columns 'start', 'end' under struct 'window' 
    words.word
).count().orderBy('window')

# Start running the query that prints the windowed word counts to the console
query = windowedCounts\
    .writeStream\
    .outputMode('complete')\
    .format('console')\
    .option('truncate', 'false')\
    .start()

query.awaitTermination()

##Spark - Streaming - few ERRORs
##ERROR-1
WARN BlockManager: Block input-0-1492690636600 replicated to only 0 peer(s) instead of 1 peers
#input RDDs are saved in memory and replicated to two nodes for fault-tolerance.
#start more Spark workers and see if the warning is gone   
    
    
##ERROR-2
Streaming program not printing result 
Do not run Spark Streaming programs locally with master configured as local or local[1]. 
This allocates only one CPU for tasks and if a receiver is running on it, 
there is no resource left to process the received data. 
Use at least local[2] to have more cores.
    
    
   
   
   

/***** Stats    MLIB - basic, chisquare test    *****/
###Spark - MLIB- Stats -  Summary statistics from DistributedMatrix 

#Distributed  RowMatrix have stats methods 
#Other Matrix can be converted to RowMatrix 
pyspark.mllib.linalg.distributed.IndexedRowMatrix(rows, numRows=0, numCols=0)
    columnSimilarities()
        Compute all cosine similarities between columns.
    toRowMatrix()
        Convert this matrix to a RowMatrix.
pyspark.mllib.linalg.distributed.BlockMatrix(blocks, rowsPerBlock, colsPerBlock, numRows=0, numCols=0)
    toIndexedRowMatrix()
        Convert this matrix to an IndexedRowMatrix.
pyspark.mllib.linalg.distributed.CoordinateMatrix(entries, numRows=0, numCols=0)
    toRowMatrix()
        Convert this matrix to a RowMatrix.
pyspark.mllib.linalg.distributed.RowMatrix(rows, numRows=0, numCols=0)
    Basic row matrix 
    columnSimilarities(threshold=0.0)
        Compute similarities between columns of this matrix.
        >>> rows = sc.parallelize([[1, 2], [1, 5]])
        >>> mat = RowMatrix(rows)
        >>> sims = mat.columnSimilarities()
        >>> sims.entries.first().value
        0.91914503...
    computeColumnSummaryStatistics()
        Computes column-wise summary statistics.
        Returns:MultivariateStatisticalSummary object containing column-wise summary statistics. 
        >>> rows = sc.parallelize([[1, 2, 3], [4, 5, 6]])
        >>> mat = RowMatrix(rows)
        >>> colStats = mat.computeColumnSummaryStatistics()
        >>> colStats.mean()
        array([ 2.5,  3.5,  4.5])
        >>> dir(colStats)
        ['call', 'count', 'max', 'mean', 'min', 'normL1', 'normL2', 'numNonzeros', 'variance']
    computeCovariance()
        Computes the covariance matrix, treating each row as an observation.
        This cannot be computed on matrices with more than 65535 columns.
        >>> rows = sc.parallelize([[1, 2], [2, 1]])
        >>> mat = RowMatrix(rows)
        >>> mat.computeCovariance() #Column-major dense matrix.
        DenseMatrix(2, 2, [0.5, -0.5, -0.5, 0.5], 0)
    computePrincipalComponents(k)
        Computes the k principal components of the given row matrix
        This cannot be computed on matrices with more than 65535 columns.
        >>> rows = sc.parallelize([[1, 2, 3], [2, 4, 5], [3, 6, 1]])
        >>> rm = RowMatrix(rows)
        # Returns the two principal components of rm
        >>> pca = rm.computePrincipalComponents(2)
        >>> pca
        DenseMatrix(3, 2, [-0.349, -0.6981, 0.6252, -0.2796, -0.5592, -0.7805], 0)
        # Transform into new dimensions with the greatest variance.
        >>> rm.multiply(pca).rows.collect() 
        [DenseVector([0.1305, -3.7394]), DenseVector([-0.3642, -6.6983]),         DenseVector([-4.6102, -4.9745])]



###Spark - MLIB- Stats -  Statistics
#These classmethods are part of pyspark.mllib.stat.Statistics
static chiSqTest(observed, expected=None)
    observed = Vector
        H0 = both are same
        conduct Pearson’s chi-squared goodness of fit test of 
        the observed data against the expected distribution, 
        or againt the uniform distribution (by default), 
        with each category having an expected frequency of 1 / len(observed).
    observed = matrix
        H0: inputs are independent
        conduct Pearson’s independence test on the input contingency matrix, 
        which cannot contain negative entries or columns or rows that sum up to 0.
    observed = RDD of LabeledPoint
        H0: feature and label are independent
        conduct Pearson’s independence test for every feature against the label 
        across the input RDD. For each feature, the (feature, label) pairs are converted 
        into a contingency matrix for which the chi-squared statistic is computed. 
        All label and feature values must be categorical.
#Example 
>>> from pyspark.mllib.linalg import Vectors, Matrices
>>> observed = Vectors.dense([4, 6, 5])
>>> pearson = Statistics.chiSqTest(observed)
>>> print(pearson.statistic)
0.4
>>> pearson.degreesOfFreedom
2
>>> print(round(pearson.pValue, 4))
0.8187
>>> pearson.method
u'pearson'
>>> pearson.nullHypothesis
u'observed follows the same distribution as expected.'
>>> observed = Vectors.dense([21, 38, 43, 80])
>>> expected = Vectors.dense([3, 5, 7, 20])
>>> pearson = Statistics.chiSqTest(observed, expected)
>>> print(round(pearson.pValue, 4))
0.0027
>>> data = [40.0, 24.0, 29.0, 56.0, 32.0, 42.0, 31.0, 10.0, 0.0, 30.0, 15.0, 12.0]
>>> chi = Statistics.chiSqTest(Matrices.dense(3, 4, data))
>>> print(round(chi.statistic, 4))
21.9958
>>> data = [LabeledPoint(0.0, Vectors.dense([0.5, 10.0])),
            LabeledPoint(0.0, Vectors.dense([1.5, 20.0])),
            LabeledPoint(1.0, Vectors.dense([1.5, 30.0])),
            LabeledPoint(0.0, Vectors.dense([3.5, 30.0])),
            LabeledPoint(0.0, Vectors.dense([3.5, 40.0])),
            LabeledPoint(1.0, Vectors.dense([3.5, 40.0])),]
>>> rdd = sc.parallelize(data, 4)
>>> chi = Statistics.chiSqTest(rdd)
>>> print(chi[0].statistic)
0.75
>>> print(chi[1].statistic)
1.5



static colStats(rdd)
    Computes column-wise summary statistics for the input RDD[Vector].
    rdd – an RDD[Vector] for which column-wise summary statistics are to be computed. 
    Returns:MultivariateStatisticalSummary object containing column-wise summary statistics. 
#Example 
>>> from pyspark.mllib.linalg import Vectors
>>> rdd = sc.parallelize([Vectors.dense([2, 0, 0, -2]),
                        Vectors.dense([4, 5, 0,  3]),
                        Vectors.dense([6, 7, 0,  8])])
>>> cStats = Statistics.colStats(rdd)
>>> dir(cStats)
['call', 'count', 'max', 'mean', 'min', 'normL1', 'normL2', 'numNonzeros', 'variance']
>>> cStats.mean()
array([ 4.,  4.,  0.,  3.])
>>> cStats.variance()
array([  4.,  13.,   0.,  25.])
>>> cStats.count()
3
>>> cStats.numNonzeros()
array([ 3.,  2.,  0.,  3.])
>>> cStats.max()
array([ 6.,  7.,  0.,  8.])
>>> cStats.min()
array([ 2.,  0.,  0., -2.])


static corr(x, y=None, method=None)
    Compute the correlation (matrix) for the input RDD(s) using the specified method. 
    Methods currently supported: pearson (default), spearman.
    If a single RDD of Vectors is passed in, 
    a correlation matrix comparing the columns in the input RDD is returned. 
    If two RDDs of floats(x,y- same cardinality ) are passed in, a single float is returned.

#Exmaple 
>>> from pyspark.mllib.linalg import Vectors
>>> rdd = sc.parallelize([Vectors.dense([1, 0, 0, -2]), Vectors.dense([4, 5, 0, 3]),
                        Vectors.dense([6, 7, 0,  8]), Vectors.dense([9, 0, 0, 1])])
>>> pearsonCorr = Statistics.corr(rdd)
>>> print(str(pearsonCorr).replace('nan', 'NaN'))
[[ 1.          0.05564149         NaN  0.40047142]
 [ 0.05564149  1.                 NaN  0.91359586]
 [        NaN         NaN  1.                 NaN]
 [ 0.40047142  0.91359586         NaN  1.        ]]
>>> spearmanCorr = Statistics.corr(rdd, method="spearman")
>>> print(str(spearmanCorr).replace('nan', 'NaN'))
[[ 1.          0.10540926         NaN  0.4       ]
 [ 0.10540926  1.                 NaN  0.9486833 ]
 [        NaN         NaN  1.                 NaN]
 [ 0.4         0.9486833          NaN  1.        ]]

>>> x = sc.parallelize([1.0, 0.0, -2.0], 2)
>>> y = sc.parallelize([4.0, 5.0, 3.0], 2)
>>> zeros = sc.parallelize([0.0, 0.0, 0.0], 2)
>>> abs(Statistics.corr(x, y) - 0.6546537) < 1e-7
True
>>> Statistics.corr(x, y) == Statistics.corr(x, y, "pearson")
True
>>> Statistics.corr(x, y, "spearman")
0.5
>>> from math import isnan
>>> isnan(Statistics.corr(x, zeros))
True


static kolmogorovSmirnovTest(data, distName='norm', *params)
    Performs the Kolmogorov-Smirnov (KS) test for data from a continuous distribution. 
    It's a 1-sample, 2-sided test 
    H0 : the data is generated from a particular distribution.
        •data – RDD, samples from the data
        •distName – string, currently only “norm” is supported. (Normal distribution) to calculate the theoretical distribution of the data.
        •params – additional values which need to be provided for a certain distribution. If not provided, the default values are used.
    Returns: KolmogorovSmirnovTestResult object containing the test statistic, 
             degrees of freedom, p-value, the method used, and the null hypothesis.

#Example 
>>> kstest = Statistics.kolmogorovSmirnovTest
>>> data = sc.parallelize([-1.0, 0.0, 1.0])
>>> ksmodel = kstest(data, "norm")
>>> print(round(ksmodel.pValue, 3))
1.0
>>> print(round(ksmodel.statistic, 3))
0.175
>>> ksmodel.nullHypothesis
u'Sample follows theoretical distribution'
>>> data = sc.parallelize([2.0, 3.0, 4.0])
>>> ksmodel = kstest(data, "norm", 3.0, 1.0)
>>> print(round(ksmodel.pValue, 3))
1.0
>>> print(round(ksmodel.statistic, 3))
0.175




###Spark - MLIB- Stats -  KernelDensity
class pyspark.mllib.stat.KernelDensity
    Estimate probability density at required points given an RDD of samples from the population.
    estimate(points)
        Estimate the probability density at points
    setBandwidth(bandwidth)
        Set bandwidth of each sample. Defaults to 1.0
    setSample(sample)
        Set sample points from the population. Should be a RDD
#Example 
>>> kd = KernelDensity()
>>> sample = sc.parallelize([0.0, 1.0])
>>> kd.setSample(sample)
>>> kd.estimate([0.0, 1.0])
array([ 0.12938758,  0.12938758])

#The bandwidth of the kernel is a free parameter 
#which exhibits a strong influence on the resulting estimate
#When the bandwidth is 0.1 (very narrow) then the kernel density estimate is said to undersmoothed 
#default is 1 

from pyspark.mllib.stat import KernelDensity
# an RDD of sample data
data = sc.parallelize([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0])

# Construct the density estimator with the sample data and a standard deviation for the Gaussian
# kernels
kd = KernelDensity()
kd.setSample(data)
kd.setBandwidth(3.0)

# Find density estimates for the given values
densities = kd.estimate([-1.0, 2.0, 5.0])













    



###Spark - MLIB- Stats -  Random data generation

class pyspark.mllib.random.RandomRDDs
        Generator methods for creating RDDs comprised of i.i.d samples from some distribution.
        Returns RDD
        RDD.stats() gives  StatCounter object having below methods 
            merge(value),mergeStats(other),copy(),count(),mean(),sum(),min(),max(),variance(),sampleVariance(),stdev(),sampleStdev() 
    static exponentialRDD(sc, mean, size, numPartitions=None, seed=None)
        Generates an RDD comprised of i.i.d. samples from the Exponential distribution with the input mean
    static exponentialVectorRDD(sc, *a, **kw)
        Generates an RDD comprised of vectors containing i.i.d. samples drawn from the Exponential distribution with the input mean.
    static gammaRDD(sc, shape, scale, size, numPartitions=None, seed=None)
        Generates an RDD comprised of i.i.d. samples from the Gamma distribution with the input shape and scale.
    static gammaVectorRDD(sc, *a, **kw)
        Generates an RDD comprised of vectors containing i.i.d. samples drawn from the Gamma distribution.
    static logNormalRDD(sc, mean, std, size, numPartitions=None, seed=None)
        Generates an RDD comprised of i.i.d. samples from the log normal distribution with the input mean and standard distribution.
    static logNormalVectorRDD(sc, *a, **kw)
        Generates an RDD comprised of vectors containing i.i.d. samples drawn from the log normal distribution
    static normalRDD(sc, size, numPartitions=None, seed=None)
        Generates an RDD comprised of i.i.d. samples from the standard normal distribution
        To transform the distribution in the generated RDD from standard normal to some other normal N(mean, sigma^2), use RandomRDDs.normal(sc, n, p, seed) .map(lambda v: mean + sigma * v)
    static normalVectorRDD(sc, *a, **kw)
        Generates an RDD comprised of vectors containing i.i.d. samples drawn from the standard normal distribution
    static poissonRDD(sc, mean, size, numPartitions=None, seed=None)
        Generates an RDD comprised of i.i.d. samples from the Poisson distribution with the input mean.
    static poissonVectorRDD(sc, *a, **kw)
        Generates an RDD comprised of vectors containing i.i.d. samples drawn from the Poisson distribution with the input mean.
    static uniformRDD(sc, size, numPartitions=None, seed=None)
        Generates an RDD comprised of i.i.d. samples from the uniform distribution U(0.0, 1.0).
        To transform the distribution in the generated RDD from U(0.0, 1.0) to U(a, b), use RandomRDDs.uniformRDD(sc, n, p, seed) .map(lambda v: a + (b - a) * v)
    static uniformVectorRDD(sc, *a, **kw)
        Generates an RDD comprised of vectors containing i.i.d. samples drawn from the uniform distribution U(0.0, 1.0).


#Example 
>>> x = RandomRDDs.normalRDD(sc, 1000, seed=1)
>>> stats = x.stats()
>>> stats.count()
1000
>>> abs(stats.mean() - 0.0) < 0.1
True
>>> abs(stats.stdev() - 1.0) < 0.1
True

>>> import numpy as np
>>> mat = np.matrix(RandomRDDs.normalVectorRDD(sc, 100, 100, seed=1).collect())
>>> mat.shape
(100, 100)
>>> abs(mat.mean() - 0.0) < 0.1
True
>>> abs(mat.std() - 1.0) < 0.1
True
>>> mean = 100.0



###Spark - ML,MLIB - Stats -  sampling and spliting on RDD and DataFrame 

##Spark - MLIB- Stats -  sampling on RDD
#Normal sampling 
rdd.sample(withReplacement, fraction, seed=None)
    Return a sampled subset of this RDD.

>>> rdd = sc.parallelize(range(100), 4)
>>> 6 <= rdd.sample(False, 0.1, 81).count() <= 14
True

#Stratified sampling for RDD[(K,V)]
rdd.sampleByKey(withReplacement, fractions, seed=None) 
rdd.sampleByKeyExact(withReplacement, fractions, seed=None)  
    The keys can be thought of as a label and the value as a specific attribute.
    For example the key can be man or woman, or document ids, 
    and the respective values can be the list of ages of the people in the population 
    or the list of words in the documents

#Example 
# an RDD of any key value pairs
data = sc.parallelize([(1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (2, 'e'), (3, 'f')])
# specify the exact fraction desired from each key as a dictionary
fractions = {1: 0.1, 2: 0.6, 3: 0.3}
approxSample = data.sampleByKey(False, fractions)


##Spark - ML- Stats -  sampling on DF

#Normal sampling 
df.sample(withReplacement, fraction, seed=None)
    Returns a sampled subset of this DataFrame.
#Example 
df = sc.parallelize([(2, 'Alice'), (5, 'Bob')])\
        .toDF(StructType([StructField('age', IntegerType()),
                          StructField('name', StringType())]))
>>> df.sample(False, 0.5, 42).count()
2

#Stratified sampling for DF[(K,V)]
sampleBy(col, fractions, seed=None)[source]
    Returns a stratified sample without replacement based on the fraction 
    given on each stratum.

#Example 
>>> from pyspark.sql.functions import col
>>> dataset = spark.range(0, 100).select((col("id") % 3).alias("key"), "id")
>>> sampled = dataset.sampleBy("key", fractions={0: 0.1, 1: 0.2}, seed=0)
>>> sampled.groupBy("key").count().orderBy("key").show()
+---+-----+
|key|count|
+---+-----+
|  0|    5|
|  1|    9|
+---+-----+

##Spark - ML, MLIB- Stats -  spliting  on RDD and DF 

rdd.randomSplit(weights, seed=None)
    Randomly splits this RDD with the provided weights.
#Example 
>>> rdd = sc.parallelize(range(500), 1)
>>> rdd1, rdd2 = rdd.randomSplit([2, 3], 17)
>>> len(rdd1.collect() + rdd2.collect())
500
>>> 150 < rdd1.count() < 250
True
>>> 250 < rdd2.count() < 350
True
        
df.randomSplit(weights, seed=None)
    Randomly splits this DataFrame with the provided weights.
#Example 
df4 = sc.parallelize([Row(course="dotNET", year=2012, earnings=10000),
                           Row(course="Java",   year=2012, earnings=20000),
                           Row(course="dotNET", year=2012, earnings=5000),
                           Row(course="dotNET", year=2013, earnings=48000),
                           Row(course="Java",   year=2013, earnings=30000)]).toDF()

>>> splits = df4.randomSplit([1.0, 2.0], 24)
>>> splits[0].count()
1
>>> splits[1].count()
3
#Example 
df = sc.parallelize(map(lambda e:Row(value=e),range(10000))).toDF()
df1,df2,df3,df4,df5 = df.randomSplit([1.0,1.0,1.0,1.0,1.0]) #spliting into five 




###Saprk -  Difference between ML and Mlib 
#ML
•New
•Pipelines
•Dataframes
•Easier to construct a practical machine learning pipeline

#MLlib
•Old
•RDD
•More features





###Spark -  General flow of classification, regression 
#ML provides a uniform set of high-level APIs built on top of DataFrame for ML 
#all methods take DF and return DF if required 

DataFrame
    ML API uses DataFrame from Spark SQL as an ML dataset, which can hold a variety of data types. 
    E.g., a DataFrame could have different columns storing text, feature vectors, true labels, and predictions.
Transformer
    A Transformer is an algorithm which can transform one DataFrame into another DataFrame. 
    E.g., an ML model is a Transformer which transforms a DataFrame with features into a DataFrame with predictions.
Estimator
    An Estimator is an algorithm which can be fit on a DataFrame to produce a Transformer. 
    E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model.
Pipeline
    A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow.
Parameter
    All Transformers and Estimators share a common API for specifying parameters.


##Steps 
1. Feature Extract, then Selection and then Transformers to convert input data, 
   model=CLASS().fit(df)  #pure transformers do not have this step 
   new_transformed_df= model.transform(DF)
   input data is 
    ML: DataFrame with columns designated by parameter inputCol or inputCols
    MLIB: Local or Distributed Vector, Matrix 
   Output data:
    ML: DataFrame with columns designated by parameter outputCol or outputCols
    MLIB: transformed Local or Distributed Vector, Matrix 
   Note data  could be dense or sparse Vector 
   sparse vector is identified by [# of_features, [indices], [corresponding_values]]
2. Random split input data into training and test by  .randomSplit(..) 
3. Instantiate Classification or regression Estimator class, say instance
4. Fit of training data to get 'model' , model=instance.fit(data)
5. predictions = model.transform(testData) to get prediction 
   in MLIB, predictions = model.predict(testData)
   input data is 
    ML: DataFrame with columns designated by parameter labelCol and featuresCol
    MLIB: list of LabeledPoint ie (label, features_vector)
   Output data:
    ML: DataFrame with columns designated by parameter predictionCol , probabilityCol
    MLIB: prediction vectors 
   features_vector, featuresCol features columns - could be dense or sparse Vector 
   sparse vector is identified by [# of_features, [indices], [corresponding_values]]
6. Evaluate :
   ML:Use instance.evaluate(predictions) or instance.evaluate(predictions, {instance.metricName: "accuracy"}) from 
       BinaryClassificationEvaluator(predictionCol="prediction")
       MulticlassClassificationEvaluator(predictionCol="prediction")
       RegressionEvaluator(predictionCol="prediction")
      MulticlassClassificationEvaluator supports metricName as "f1" (default), "weightedPrecision", "weightedRecall", "accuracy"
      BinaryClassificationEvaluator supports "areaUnderROC" (default), "areaUnderPR")
      RegressionEvaluator supports "rmse" (default): root mean squared error,"mse": mean squared error,"r2": R2 metric,"mae": mean absolute error 
   MLIB, similarly use  Metrics classes 
7. Note for Binary classification or regression  , 
   can get summary by model.summary 
        BinaryLogisticRegressionSummary, LinearRegressionSummary, GeneralizedLinearRegressionSummary 
   BinaryLogisticRegressionSummary have summary.predictions(a DF with columns ['label', 'features', 'rawPrediction', 'probability', 'prediction'])
   RegressionSummary have summary.pValues, 
    summary.coefficientStandardErrors(form normal solver), summary.residuals, summary.aic (for GLM)

#Optional STeps 
1. Create a Pipeline (Optional) having Transformer, Estimator etc which can be fit() and transform() together 
2. Use CrossValidation  for parameter tunining to get best Model 





###Spark - ML - Parameter 
#Param and Params represent a parameter or parameters of a Transformer, Estimator etc 

#Example - Standard Input columns in input DataFrame for Estimator 
#Param name  Type(s)     Default column name      Description
labelCol    Double      "label"                     Label to predict 
featuresCol Vector      "features"                  Feature vector 

#Example - Standard Output columns in output DataFrame from a fitted Model
#Param name          Type(s)     Default column name         Description
predictionCol       Double      "prediction"                 Predicted label  
rawPredictionCol    Vector      "rawPrediction"              Vector of length # classes, with the counts of training instance labels at the tree node which makes the prediction Classification only 
probabilityCol      Vector      "probability"                Vector of length # classes equal to rawPrediction normalized to a multinomial distribution Classification only 
varianceCol         Double                                   The biased sample variance of prediction Regression only 

##Usages 
#For example - Estimator - LogisticRegression()
#Fitting:  instance.fit() on labelCol and featuresCol of training data to get model
model = estimator.fit(training)
#Predictions: input: 'featuresCol' of test data , output: predictionCol, probabilityCol of result DF 
predictions = model.transform(test)


##Reference 
class pyspark.ml.param.Param(parent, name, doc, typeConverter=None)
    A param with self-contained documentation.

class pyspark.ml.param.Params
    Components that take parameters. 
    This also provides an internal param map to store parameter values 
    attached to the instance.
    copy(extra=None)
        Creates a copy of this instance with the same uid and some extra params. 
    explainParam(param)
        Explains a single param and returns its name, doc, and optional default value and user-supplied value in a string.
    explainParams()
        Returns the documentation of all params with their optionally default values and user-supplied values.
    extractParamMap(extra=None)
        Extracts the param values with extra values from input into a flat param map, 
        where the latter value is used if there exist conflicts, 
        i.e., with ordering: default param values < user-supplied values < extra.
        Returns: merged param map 
    getOrDefault(param)
        Gets the value of a param in the user-supplied param map or its default value. 
        Raises an error if neither is set.
    getParam(paramName)
        Gets a param by its name.
    hasDefault(param)
        Checks whether a param has a default value.
    hasParam(paramName)
        Tests whether this instance contains a param with a given (string) name.
    isDefined(param)
        Checks whether a param is explicitly set by user or has a default value.
    isSet(param)
        Checks whether a param is explicitly set by user.
    params
        Returns all params ordered by name. 
 

##List of all Parameters used in ML  
#(paramName, documentation, default_value, converter)
#default value is DF_Column_name for some parameters , None means no default 
shared = [
        ("maxIter", "max number of iterations (>= 0).", None, "TypeConverters.toInt"),
        ("regParam", "regularization parameter (>= 0).", None, "TypeConverters.toFloat"),
        ("featuresCol", "features column name.", "'features'", "TypeConverters.toString"),
        ("labelCol", "label column name.", "'label'", "TypeConverters.toString"),
        ("predictionCol", "prediction column name.", "'prediction'", "TypeConverters.toString"),
        ("probabilityCol", "Column name for predicted class conditional probabilities. " +
         "Note: Not all models output well-calibrated probability estimates! These probabilities " +
         "should be treated as confidences, not precise probabilities.", "'probability'",  "TypeConverters.toString"),
        ("rawPredictionCol", "raw prediction (a.k.a. confidence) column name.", "'rawPrediction'",
         "TypeConverters.toString"),
        ("inputCol", "input column name.", None, "TypeConverters.toString"),
        ("inputCols", "input column names.", None, "TypeConverters.toListString"),
        ("outputCol", "output column name.", "self.uid + '__output'", "TypeConverters.toString"),
        ("outputCols", "output column names.", None, "TypeConverters.toListString"),
        ("numFeatures", "number of features.", None, "TypeConverters.toInt"),
        ("checkpointInterval", "set checkpoint interval (>= 1) or disable checkpoint (-1). " +
         "E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: " +
         "this setting will be ignored if the checkpoint directory is not set in the SparkContext.",
         None, "TypeConverters.toInt"),
        ("seed", "random seed.", "hash(type(self).__name__)", "TypeConverters.toInt"),
        ("tol", "the convergence tolerance for iterative algorithms (>= 0).", None,
         "TypeConverters.toFloat"),
        ("stepSize", "Step size to be used for each iteration of optimization (>= 0).", None,
         "TypeConverters.toFloat"),
        ("handleInvalid", "how to handle invalid entries. Options are skip (which will filter " +
         "out rows with bad values), or error (which will throw an error). More options may be " +
         "added later.", None, "TypeConverters.toString"),
        ("elasticNetParam", "the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, " +
         "the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.", "0.0",
         "TypeConverters.toFloat"),
        ("fitIntercept", "whether to fit an intercept term.", "True", "TypeConverters.toBoolean"),
        ("standardization", "whether to standardize the training features before fitting the " +
         "model.", "True", "TypeConverters.toBoolean"),
        ("thresholds", "Thresholds in multi-class classification to adjust the probability of " +
         "predicting each class. Array must have length equal to the number of classes, with " +
         "values > 0, excepting that at most one value may be 0. " +
         "The class with largest value p/t is predicted, where p is the original " +
         "probability of that class and t is the class's threshold.", None,
         "TypeConverters.toListFloat"),
        ("threshold", "threshold in binary classification prediction, in range [0, 1]",
         "0.5", "TypeConverters.toFloat"),
        ("weightCol", "weight column name. If this is not set or empty, we treat " +
         "all instance weights as 1.0.", None, "TypeConverters.toString"),
        ("solver", "the solver algorithm for optimization. If this is not set or empty, " +
         "default value is 'auto'.", "'auto'", "TypeConverters.toString"),
        ("varianceCol", "column name for the biased sample variance of prediction.",
         None, "TypeConverters.toString"),
        ("aggregationDepth", "suggested depth for treeAggregate (>= 2).", "2",
         "TypeConverters.toInt"),
        ("parallelism", "the number of threads to use when running parallel algorithms (>= 1).",
         "1", "TypeConverters.toInt"),
        ("loss", "the loss function to be optimized.", None, "TypeConverters.toString")]

decisionTreeParams = [
#(paramName, documentation, converter)
        ("maxDepth", "Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; " +
         "depth 1 means 1 internal node + 2 leaf nodes.", "TypeConverters.toInt"),
        ("maxBins", "Max number of bins for" +
         " discretizing continuous features.  Must be >=2 and >= number of categories for any" +
         " categorical feature.", "TypeConverters.toInt"),
        ("minInstancesPerNode", "Minimum number of instances each child must have after split. " +
         "If a split causes the left or right child to have fewer than minInstancesPerNode, the " +
         "split will be discarded as invalid. Should be >= 1.", "TypeConverters.toInt"),
        ("minInfoGain", "Minimum information gain for a split to be considered at a tree node.",
         "TypeConverters.toFloat"),
        ("maxMemoryInMB", "Maximum memory in MB allocated to histogram aggregation. If too small," +
         " then 1 node will be split per iteration, and its aggregates may exceed this size.",
         "TypeConverters.toInt"),
        ("cacheNodeIds", "If false, the algorithm will pass trees to executors to match " +
         "instances with nodes. If true, the algorithm will cache node IDs for each instance. " +
         "Caching can speed up training of deeper trees. Users can set how often should the " +
         "cache be checkpointed or disable it by setting checkpointInterval.",
         "TypeConverters.toBoolean")]

    


##Example with LogisticRegression
class LogisticRegression(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasMaxIter,
                         HasRegParam, HasTol, HasProbabilityCol, HasRawPredictionCol,
                         HasElasticNetParam, HasFitIntercept, HasStandardization, HasThresholds,
                         HasWeightCol, HasAggregationDepth, JavaMLWritable, JavaMLReadable)


#All these mixin are derived from Params 

#Note for DF column name mixin , if suffx = 'sCol' means that multiple columns 
#If suffix='Col', then it is single column  

#Each parameter mixin defines one parameter and has below attributes
#string of Param = paramName - "lowercase of Mixin class name after stripping 'Has'"
#this paramName is required for getParam(paramName)
    parameter_variable_name = lowercase of Mixin class name after stripping 'Has'
    set*(value)
        to set the value of that parameter 
        * = Mixin class name after stripping 'Has'
    get*()
        Get the current value 
        * = Mixin class name after stripping 'Has'
#for example 
class HasLabelCol(Params)
    labelCol = Param(Params._dummy(), "labelCol", "label column name.", typeConverter=TypeConverters.toString)
    setLabelCol(value)
    getLabelCol()
class HasFeaturesCol(Params)
    featuresCol = Param(Params._dummy(), "featuresCol", "features column name.", typeConverter=TypeConverters.toString)
    setFeaturesCol(value)
    getFeaturesCol()
    
    
    
#Example code 
from __future__ import print_function


from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("EstimatorTransformerParamExample")\
    .getOrCreate()

# Prepare training data from a list of (label, features) tuples.
training = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])
    


# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.01)

>>> p = lr.getParam('labelCol')
>>> p
Param(parent='LogisticRegression_4b378ab90cbb69bdce34', name='labelCol', doc='label column name.')

>>> lr.explainParam(p)
'labelCol: label column name. (default: label)'

>>> lr.getOrDefault(p)
'label'

>>> training.show()  #Note columns Name 
+-----+--------------+
|label|      features|
+-----+--------------+
|  1.0| [0.0,1.1,0.1]|
|  0.0|[2.0,1.0,-1.0]|
|  0.0| [2.0,1.3,1.0]|
|  1.0|[0.0,1.2,-0.5]|
+-----+--------------+


>>> print(lr.explainParams()) #note the default and current value , undefined means no current value 
aggregationDepth: suggested depth for treeAggregate (>= 2). (default: 2)
elasticNetParam: the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty. (default: 0.0)
family: The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial (default: auto)
featuresCol: features column name. (default: features)
fitIntercept: whether to fit an intercept term. (default: True)
labelCol: label column name. (default: label)
maxIter: max number of iterations (>= 0). (default: 100, current: 10)
predictionCol: prediction column name. (default: prediction)
probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities. (default: probability)
rawPredictionCol: raw prediction (a.k.a. confidence) column name. (default: rawPrediction)
regParam: regularization parameter (>= 0). (default: 0.0, current: 0.01)
standardization: whether to standardize the training features before fitting the model. (default: True)
threshold: Threshold in binary classification prediction, in range [0, 1]. If threshold and thresholds are both set, they must match.e.g. if threshold is p, the n thresholds must be equal to [1-p, p]. (default: 0.5)
thresholds: Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes,with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold. (undefined)
tol: the convergence tolerance for iterative algorithms (>= 0). (default: 1e-06)
weightCol: weight column name. If this is not set or empty, we treat all instance weights as 1.0. (undefined)

>>> lr.params
[Param(parent='LogisticRegression_4b378ab90cbb69bdce34', name='aggregationDepth'
, doc='suggested depth for treeAggregate (>= 2).'), ....]

>>> lr.extractParamMap() #{ Param_instance:current_value , ...}
{Param(parent='LogisticRegression_4b378ab90cbb69bdce34', name='maxIter', 
 doc='max number of iterations (>= 0).'): 10, ...}
 
>>> {k.name:v for k,v in lr.extractParamMap().items()}
{'family': 'auto', 'elasticNetParam': 0.0, 'probabilityCol': 'probability', 
'fitIntercept': True, 'labelCol': 'label', 'featuresCol': 'features', 'tol': 1e-06,
'threshold': 0.5, 'standardization': True, 'aggregationDepth': 2, 
'predictionCol': 'prediction', 'regParam': 0.01, 'maxIter': 10, 
'rawPredictionCol': 'rawPrediction'}
 
#get/set method for parameters 
>>> [print(n) for n in sorted(dir(lr)) if n.startswith("set") or n.startswith("get")]
getAggregationDepth
getElasticNetParam
getFamily
getFeaturesCol
getFitIntercept
getLabelCol
getMaxIter
getOrDefault
getParam
getPredictionCol
getProbabilityCol
getRawPredictionCol
getRegParam
getStandardization
getThreshold
getThresholds
getTol
getWeightCol
setAggregationDepth
setElasticNetParam
setFamily
setFeaturesCol
setFitIntercept
setLabelCol
setMaxIter
setParams
setPredictionCol
setProbabilityCol
setRawPredictionCol
setRegParam
setStandardization
setThreshold
setThresholds
setTol
setWeightCol


# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(training)
>>> model1.summary.predictions.show()
+-----+--------------+--------------------+--------------------+----------+
|label|      features|       rawPrediction|         probability|prediction|
+-----+--------------+--------------------+--------------------+----------+
|  1.0| [0.0,1.1,0.1]|[-2.8991948946380...|[0.05219337666300...|       1.0|
|  0.0|[2.0,1.0,-1.0]|[3.14530074643784...|[0.95872315828999...|       0.0|
|  0.0| [2.0,1.3,1.0]|[3.12319457002747...|[0.95783942352957...|       0.0|
|  1.0|[0.0,1.2,-0.5]|[-3.3881238419959...|[0.03266869266264...|       1.0|
+-----+--------------+--------------------+--------------------+----------+

# We may alternatively specify parameters using a Python dictionary as a paramMap
paramMap = {lr.maxIter: 20}
paramMap[lr.maxIter] = 30  # Specify 1 Param, overwriting the original maxIter.
paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  # Specify multiple Params.

# You can combine paramMaps, which are python dictionaries.
paramMap2 = {lr.probabilityCol: "myProbability"}  # Change output column name
paramMapCombined = paramMap.copy()
paramMapCombined.update(paramMap2)

# Now learn a new model using the paramMapCombined parameters.
# paramMapCombined overrides all parameters set earlier via lr.set* methods.
model2 = lr.fit(training, paramMapCombined)

>>> model2.numFeatures
3
>>> model2.numClasses
2
>>> model2.coefficients  #coeff for each feature 
DenseVector([-1.4314, 0.4321, -0.1492])
>>> model2.intercept
0.9191234482945521
>>> model2.interceptVector
DenseVector([0.9191])

# Prepare test data
test = spark.createDataFrame([
    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
    (0.0, Vectors.dense([3.0, 2.0, -0.1])),
    (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

# Make predictions on test data using the Transformer.transform() method.
# LogisticRegression.transform will only use the 'features' column.
# Note that model2.transform() outputs a "myProbability" column instead of the usual
# 'probability' column since we renamed the lr.probabilityCol parameter previously.
prediction = model2.transform(test) #DT 
>>> prediction.show()
+-----+--------------+--------------------+--------------------+----------+
|label|      features|       rawPrediction|       myProbability|prediction|
+-----+--------------+--------------------+--------------------+----------+
|  1.0|[-1.0,1.5,1.3]|[-2.8046569418746...|[0.05707304171034...|       1.0|
|  0.0|[3.0,2.0,-0.1]|[2.49587635664210...|[0.92385223117041...|       0.0|
|  1.0|[0.0,2.2,-1.5]|[-2.0935249027913...|[0.10972776114780...|       1.0|
+-----+--------------+--------------------+--------------------+----------+


spark.stop()
   







###Spark - ML - Feature extraction, seletion and Tranformation 
#Note Transformer, Estimator are inherited from Params , and has Param methods 
#Model is derived from Transformer 


class pyspark.ml.Model(Transformer)
    Abstract class for models that are fitted by estimators.
    explainParams()
        Returns the documentation of all params with their optionally default values and user-supplied values.
    extractParamMap(extra=None)
        Extracts the embedded default param values and user-supplied values, and then merges them with extra values from input into a flat param map, where the latter value is used if there exist conflicts, i.e., with ordering: default param values < user-supplied values < extra.
    transform(dataset, params=None)
        predicts the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params.
    Returns:transformed DF
    
    
class pyspark.ml.Transformer(Params)
    Abstract class for transformers that transform one dataset into another.
    explainParams()
        Returns the documentation of all params with their optionally default values and user-supplied values.
    extractParamMap(extra=None)
        Extracts the embedded default param values and user-supplied values, and then merges them with extra values from input into a flat param map, where the latter value is used if there exist conflicts, i.e., with ordering: default param values < user-supplied values < extra.
    transform(dataset, params=None)
        Transforms the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params.
    Returns:transformed DF


class pyspark.ml.Estimator(Params)
Abstract class for estimators that fit models to data.
    explainParams()
        Returns the documentation of all params with their optionally default values and user-supplied values.
    extractParamMap(extra=None)
        Extracts the embedded default param values and user-supplied values, and then merges them with extra values from input into a flat param map, where the latter value is used if there exist conflicts, i.e., with ordering: default param values < user-supplied values < extra.
    fit(dataset, params=None)
        Fits a model to the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params. 
            If a list/tuple of param maps is given, this calls fit on each param map and returns a list of models.
    Returns: Model
 

##Few Important  mixin used 
#Note for DF column 'Col' name mixin , if suffx = 'sCol' means that multiple columns 
#If suffix='Col', then it is single column  

#Each parameter mixin defines one parameter and has below attributes
#string of Param = paramName - "lowercase of Mixin class name after stripping 'Has'"
#this paramName is required for getParam(paramName)
    parameter_variable_name = lowercase of Mixin class name after stripping 'Has'
    set*(value)
        to set the value of that parameter 
        * = Mixin class name after stripping 'Has'
    get*()
        Get the current value 
        * = Mixin class name after stripping 'Has'
class HasInputCol(Params)
    inputCol = Param(Params._dummy(), "inputCol", "input column name.", typeConverter=TypeConverters.toString)
    setInputCol(value)
    getInputCol()
class HasInputCols(Params)
    inputCols = Param(Params._dummy(), "inputCols", "input column names.", typeConverter=TypeConverters.toListString)
    setInputCols(value)
    getInputCols()
class HasOutputCol(Params)
    outputCol = Param(Params._dummy(), "outputCol", "output column name.", typeConverter=TypeConverters.toString)
    setOutputCol(value)
    getOutputCol()
class HasOutputCols(Params):
    outputCols = Param(Params._dummy(), "outputCols", "output column names.", typeConverter=TypeConverters.toListString)
    setOutputCols(value)
    getOutputCols()
    
#JavaMLWritable exposes interface .write().save(path)
#JavaMLReadable exposes classmethod interface .read().load(path) , Returns a instance 
class JavaMLWritable(MLWritable):
    write()
        Returns an JavaMLWriter instance for this ML instance
class JavaMLWriter(MLWriter)
    save(path):
        Save the ML instance to the input path
    overwrite(self)
        Overwrites if the output path already exists
class JavaMLReadable(MLReadable):
    classmethod read():
        Returns an JavaMLReader instance for this class
class JavaMLReader(MLReader):
    load(path):
        Load the ML instance from the input path
#Transformer, estimator       
JavaTransformer
    This mixin makes the class pure Tranformer (pyspark.ml.Transformer )
    use as instance.transform(df) to get transformed data 
JavaEstimator
    This mixin makes the class Estimator (pyspark.ml.Estimator )
    use as instance.fit(df).transform(df) to get transformed data 
    ie .fit(df) would return a Model and Model has .transform(df) 
        
        
        
##Spark - feature - Reference  - Trandformer Based  
class Binarizer(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable):
    Binarize(0/1) a column of continuous features given a threshold.
    Mixin:
        JavaTransformer = has functionality of pyspark.ml.Transformer ie transform 
        HasInputCol = one input column, default paramName 'inputCol'
        HasOutputCol = one output column , default paramName 'outputCol'
        JavaMLReadable = exposes classmethod interface .read().load(path)  
        JavaMLWritable = exposes interface .write().save(path)
    Has all methods of Params , mainly 
        explainParams()
            Returns the documentation of all params with their optionally default values and user-supplied values.
        extractParamMap(extra=None)
            Extracts the embedded default param values and user-supplied values, and then merges them with extra values from input into a flat param map, where the latter value is used if there exist conflicts, i.e., with ordering: default param values < user-supplied values < extra.
    Has set*/get* methods from Param mixing eg 
        getInputCol()
        getOutputCol()
        getThreshold()
        setInputCol(value)
        setOutputCol(value)
        setThreshold(value)
        setParam(**params)
        getParam(paramName)
    Has methods of JavaMLReadable, JavaMLWritable
        read()
        write()
        save(path)
            shortcut of .write().save(path)
        load(path) 
            shortcut of .read().load(path)  
    
#Example 
df = spark.createDataFrame([(0.5,)], ["values"])
binarizer = Binarizer(threshold=1.0, inputCol="values", outputCol="features")
>>> binarizer.transform(df).show()
+------+--------+
|values|features|
+------+--------+
|   0.5|     0.0|
+------+--------+
>>> binarizer.transform(df).head().features
0.0
>>> binarizer.setParams(outputCol="freqs").transform(df).head().freqs
0.0

params = {binarizer.threshold: -0.5, binarizer.outputCol: "vector"}
>>> binarizer.transform(df, params).head().vector
1.0

binarizerPath = temp_path + "/binarizer"
binarizer.save(binarizerPath)
loadedBinarizer = Binarizer.load(binarizerPath)
>>> loadedBinarizer.getThreshold() == binarizer.getThreshold()
True

>>> binarizer.getParam('outputCol')
Param(parent='Binarizer_4f5d96a6f4223583c13d', name='outputCol', doc='output col
umn name.')
>>> {k.name:v for k,v in binarizer.extractParamMap().items()}
{'threshold': 1.0, 'inputCol': 'values', 'outputCol': 'features'}

>>> binarizer.params
[Param(parent='Binarizer_4f5d96a6f4223583c13d', name='inputCol', doc='input colu
mn name.'), Param(parent='Binarizer_4f5d96a6f4223583c13d', name='outputCol', doc
='output column name.'), Param(parent='Binarizer_4f5d96a6f4223583c13d', name='th
reshold', doc='threshold in binary classification prediction, in range [0, 1]')]

>>> binarizer.extractParamMap()
{Param(parent='Binarizer_4f5d96a6f4223583c13d', name='outputCol', doc='output co
lumn name.'): 'features', Param(parent='Binarizer_4f5d96a6f4223583c13d', name='t
hreshold', doc='threshold in binary classification prediction, in range [0, 1]')
: 1.0, Param(parent='Binarizer_4f5d96a6f4223583c13d', name='inputCol', doc='inpu
t column name.'): 'values'}

>>> print(binarizer.explainParams())
inputCol: input column name. (current: values)
outputCol: output column name. (default: Binarizer_4f5d96a6f4223583c13d__output, current: features)
threshold: threshold in binary classification prediction, in range [0, 1] (default: 0.0, current: 1.0)

##Another example - with HasInputCols - multiple input cols 
VectorAssembler(JavaTransformer, HasInputCols, HasOutputCol, JavaMLReadable, JavaMLWritable)
    A feature transformer that merges multiple columns into a vector column.
    
#code  
df = spark.createDataFrame([(1, 0, 3)], ["a", "b", "c"])
vecAssembler = VectorAssembler(inputCols=["a", "b", "c"], outputCol="features")
>>> vecAssembler.transform(df).head().features
DenseVector([1.0, 0.0, 3.0])
>>> vecAssembler.setParams(outputCol="freqs").transform(df).head().freqs
DenseVector([1.0, 0.0, 3.0])
>>> params = {vecAssembler.inputCols: ["b", "a"], vecAssembler.outputCol: "vector"}
>>> vecAssembler.transform(df, params).head().vector
DenseVector([0.0, 1.0])
>>> vectorAssemblerPath = temp_path + "/vector-assembler"
>>> vecAssembler.save(vectorAssemblerPath)
>>> loadedAssembler = VectorAssembler.load(vectorAssemblerPath)
>>> loadedAssembler.transform(df).head().freqs == vecAssembler.transform(df).head().freqs
True


##Other transformer based classes 
Bucketizer(JavaTransformer, HasInputCol, HasOutputCol, HasHandleInvalid,JavaMLReadable, JavaMLWritable)        
DCT(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
ElementwiseProduct(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable,JavaMLWritable)
HashingTF(JavaTransformer, HasInputCol, HasOutputCol, HasNumFeatures, JavaMLReadable,JavaMLWritable)
NGram(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
Normalizer(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
OneHotEncoder(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
PolynomialExpansion(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable,JavaMLWritable)
RegexTokenizer(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
SQLTransformer(JavaTransformer, JavaMLReadable, JavaMLWritable)
IndexToString(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
StopWordsRemover(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
Tokenizer(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
VectorSlicer(JavaTransformer, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
#Note below is with HasInputCols, hence with multiple columns 
FeatureHasher(JavaTransformer, HasInputCols, HasOutputCol, HasNumFeatures, JavaMLReadable,JavaMLWritable)



##Spark - feature - Reference  - Estimator based  
PCA(JavaEstimator, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
    PCA trains a model to project vectors to a lower dimensional space of the top k principal components.
    Mixin:
        JavaEstimator = has functionality of pyspark.ml.Estimator ie model = fit(df) and them model.transform(df)
        HasInputCol = one input column, default paramName 'inputCol'
        HasOutputCol = one output column , default paramName 'outputCol'
        JavaMLReadable = exposes classmethod interface .read().load(path)  
        JavaMLWritable = exposes interface .write().save(path)
    Has all methods of Params , mainly 
        explainParams()
            Returns the documentation of all params with their optionally default values and user-supplied values.
        extractParamMap(extra=None)
            Extracts the embedded default param values and user-supplied values, and then merges them with extra values from input into a flat param map, where the latter value is used if there exist conflicts, i.e., with ordering: default param values < user-supplied values < extra.
    Has set*/get* methods from Param mixing eg 
        getInputCol()
        getOutputCol()
        getThreshold()
        setInputCol(value)
        setOutputCol(value)
        setThreshold(value)
        setParam(**params)
        getParam(paramName)
    Has methods of JavaMLReadable, JavaMLWritable
        read()
        write()
        save(path)
            shortcut of .write().save(path)
        load(path) 
            shortcut of .read().load(path)  
#Example 
from pyspark.ml.linalg import Vectors
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
    (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
    (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data,["features"])
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(df)
>>> model.transform(df).collect()[0].pca_features
DenseVector([1.648..., -4.013...])
>>> model.explainedVariance
DenseVector([0.794..., 0.205...])

pcaPath = temp_path + "/pca"
pca.save(pcaPath)
loadedPca = PCA.load(pcaPath)
>>> loadedPca.getK() == pca.getK()
True
>>> modelPath = temp_path + "/pca-model"
>>> model.save(modelPath)
>>> loadedModel = PCAModel.load(modelPath)
>>> loadedModel.pc == model.pc
True
>>> loadedModel.explainedVariance == model.explainedVariance
True

>>> pca.getParam('outputCol')
Param(parent='PCA_48aca19c24f274c84253', name='outputCol', doc='output column name.')

>>> {k.name:v for k,v in pca.extractParamMap().items()}
{'k': 2, 'inputCol': 'features', 'outputCol': 'pca_features'}

>>> pca.params
[Param(parent='PCA_48aca19c24f274c84253', name='inputCol', doc='input column nam
e.'), Param(parent='PCA_48aca19c24f274c84253', name='k', doc='the number of prin
cipal components'), Param(parent='PCA_48aca19c24f274c84253', name='outputCol', d
oc='output column name.')]

>>> pca.extractParamMap()
{Param(parent='PCA_48aca19c24f274c84253', name='outputCol', doc='output column n
ame.'): 'pca_features', Param(parent='PCA_48aca19c24f274c84253', name='k', doc='
the number of principal components'): 2, Param(parent='PCA_48aca19c24f274c84253'
, name='inputCol', doc='input column name.'): 'features'}

>>> print(pca.explainParams())
inputCol: input column name. (current: features)
k: the number of principal components (current: 2)
outputCol: output column name. (default: PCA_48aca19c24f274c84253__output, current: pca_features)


##Another example - with HasInputCols - multiple input cols 
Imputer(JavaEstimator, HasInputCols, JavaMLReadable, JavaMLWritable)
    Imputation estimator for completing missing values, either using the mean or the median of the columns in which the missing values are located

#code 
df = spark.createDataFrame([(1.0, float("nan")), (2.0, float("nan")), (float("nan"), 3.0),
                                (4.0, 4.0), (5.0, 5.0)], ["a", "b"])
imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])
model = imputer.fit(df)
>>> model.surrogateDF.show()
+---+---+
|  a|  b|
+---+---+
|3.0|4.0|
+---+---+
>>> model.transform(df).show()
+---+---+-----+-----+
|  a|  b|out_a|out_b|
+---+---+-----+-----+
|1.0|NaN|  1.0|  4.0|
|2.0|NaN|  2.0|  4.0|
|NaN|3.0|  3.0|  3.0|
...
>>> imputer.setStrategy("median").setMissingValue(1.0).fit(df).transform(df).show()
+---+---+-----+-----+
|  a|  b|out_a|out_b|
+---+---+-----+-----+
|1.0|NaN|  4.0|  NaN|

imputerPath = temp_path + "/imputer"
imputer.save(imputerPath)
loadedImputer = Imputer.load(imputerPath)
>>> loadedImputer.getStrategy() == imputer.getStrategy()
True
>>> loadedImputer.getMissingValue()
1.0

modelPath = temp_path + "/imputer-model"
model.save(modelPath)
loadedModel = ImputerModel.load(modelPath)
>>> loadedModel.transform(df).head().out_a == model.transform(df).head().out_a
True




##Other Estimator based classes , 
BucketedRandomProjectionLSH(JavaEstimator, LSHParams, HasInputCol, HasOutputCol, HasSeed,JavaMLReadable, JavaMLWritable)
CountVectorizer(JavaEstimator, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
IDF(JavaEstimator, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
MaxAbsScaler(JavaEstimator, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
MinHashLSH(JavaEstimator, LSHParams, HasInputCol, HasOutputCol, HasSeed,JavaMLReadable, JavaMLWritable)
MinMaxScaler(JavaEstimator, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
QuantileDiscretizer(JavaEstimator, HasInputCol, HasOutputCol, HasHandleInvalid,JavaMLReadable, JavaMLWritable)
StandardScaler(JavaEstimator, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable)
StringIndexer(JavaEstimator, HasInputCol, HasOutputCol, HasHandleInvalid, JavaMLReadable,
VectorIndexer(JavaEstimator, HasInputCol, HasOutputCol, HasHandleInvalid, JavaMLReadable,JavaMLWritable)
Word2Vec(JavaEstimator, HasStepSize, HasMaxIter, HasSeed, HasInputCol, HasOutputCol,, JavaMLReadable, JavaMLWritable)
RFormula(JavaEstimator, HasFeaturesCol, HasLabelCol, HasHandleInvalid, JavaMLReadable, JavaMLWritable)
ChiSqSelector(JavaEstimator, HasFeaturesCol, HasOutputCol, HasLabelCol, JavaMLReadable,JavaMLWritable)


 
 
 
##Spark - ML - Extracting, transforming and selecting features- Quick Intro 
#Note almost all  have HasInputCol, HasOutputCol mixins 
#Hence one input/output column(which might be Vector)
#If mixin is HasInputCols, then multiple input columns 
•Extraction:        Extracting features from 'raw' data
•Transformation:    Scaling, converting, or modifying features
•Selection:         Selecting a subset from a larger set of features

#inputCol and outputCol can be pyspark.ml.linalg.DenseVector  or pyspark.ml.linalg.SparseVector 
#DenseVector format - [value1, value2,...]
#Sparse Vector format - (# of_features, [indices], [corresponding_values])



##Spark - ML - QuickSummary - Feature Extractors , use with pyspark.ml.feature 
◦TF-IDF     
    Term frequency-inverse document frequency (TF-IDF) is 
    multiplication of TF(HashingTF and CountVectorizer) and IDF 
    Reflects the importance of a term to a document 
    TF - generates the term frequency vectors
    IDF -  it down-weights columns which appear frequently in the corpus 
    #Example Result 
    #inputCol-sentence, Tokenizer outputCol - words, TF outputCol -rawfeatures, final outputCol(TF-IDF)- features
    +-----+-----------------------------------+------------------------------------------+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------+
    |label|sentence                           |words                                     |rawFeatures                              |features                                                                                                              |
    +-----+-----------------------------------+------------------------------------------+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------+
    |0.0  |Hi I heard about Spark             |[hi, i, heard, about, spark]              |(20,[0,5,9,17],[1.0,1.0,1.0,2.0])        |(20,[0,5,9,17],[0.6931471805599453,0.6931471805599453,0.28768207245178085,1.3862943611198906])                        |
    |0.0  |I wish Java could use case classes |[i, wish, java, could, use, case, classes]|(20,[2,7,9,13,15],[1.0,1.0,3.0,1.0,1.0]) |(20,[2,7,9,13,15],[0.6931471805599453,0.6931471805599453,0.8630462173553426,0.28768207245178085,0.28768207245178085]) |
    |1.0  |Logistic regression models are neat|[logistic, regression, models, are, neat] |(20,[4,6,13,15,18],[1.0,1.0,1.0,1.0,1.0])|(20,[4,6,13,15,18],[0.6931471805599453,0.6931471805599453,0.28768207245178085,0.28768207245178085,0.6931471805599453])|
    +-----+-----------------------------------+------------------------------------------+-----------------------------------------+----------------------------------------------------------------------------------------------------------------------+

◦Word2Vec
    Transforms each column of Vector into a vector using the average of all words in the document
    #Example Result 
    #inputCol-text, outputCol - result
    +------------------------------------------+----------------------------------------------------------------+
    |text                                      |result                                                          |
    +------------------------------------------+----------------------------------------------------------------+
    |[Hi, I, heard, about, Spark]              |[-0.008142343163490296,0.02051363289356232,0.03255096450448036] |
    |[I, wish, Java, could, use, case, classes]|[0.043090314205203734,0.035048123182994974,0.023512658663094044]|
    |[Logistic, regression, models, are, neat] |[0.038572299480438235,-0.03250147425569594,-0.01552378609776497]|
    +------------------------------------------+----------------------------------------------------------------+

    
◦CountVectorizer
    Converts a collection of text documents to vectors of token counts.
    #Example Result 
    #inputCol-words, outputCol - features
    +---+---------------+-------------------------+
    |id |words          |features                 |
    +---+---------------+-------------------------+
    |0  |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
    |1  |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
    +---+---------------+-------------------------+

    
    
    
##Spark - ML - QuickSummary - Feature Transformers , use with pyspark.ml.feature 
◦RegexTokenizer
    takes text (such as a sentence) and breakes it into individual terms (usually words). 
    #Example Result 
    #inputCol-sentence, outputCol - words , extra Param .setPattern("\\W"))
    +---+-----------------------------------+------------------------------------------+
    |id |sentence                           |words                                     |
    +---+-----------------------------------+------------------------------------------+
    |0  |Hi I heard about Spark             |[hi, i, heard, about, spark]              |
    |1  |I wish Java could use case classes |[i, wish, java, could, use, case, classes]|
    |2  |Logistic,regression,models,are,neat|[logistic, regression, models, are, neat] |
    +---+-----------------------------------+------------------------------------------+
    
◦StopWordsRemover
    Removes Stop words are words which should be excluded from the input, 
    typically because the words appear frequently and don't carry as much meaning.
    #Example Result 
    #inputCol-raw, outputCol - filtered 
    +---+----------------------------+--------------------+
    |id |raw                         |filtered            |
    +---+----------------------------+--------------------+
    |0  |[I, saw, the, red, balloon] |[saw, red, balloon] |
    |1  |[Mary, had, a, little, lamb]|[Mary, little, lamb]|
    +---+----------------------------+--------------------+
    
◦NGram
     transform input feature/a Column Vector into sequence of n tokens (typically words) for some integer n
     #Example Result 
     #inputCol-words, outputCol - ngrams , extra Param setN(2)
    +---+------------------------------------------+------------------------------------------------------------------+
    |id |words                                     |ngrams                                                            |
    +---+------------------------------------------+------------------------------------------------------------------+
    |0  |[Hi, I, heard, about, Spark]              |[Hi I, I heard, heard about, about Spark]                         |
    |1  |[I, wish, Java, could, use, case, classes]|[I wish, wish Java, Java could, could use, use case, case classes]|
    |2  |[Logistic, regression, models, are, neat] |[Logistic regression, regression models, models are, are neat]    |
    +---+------------------------------------------+------------------------------------------------------------------+

     

◦Binarizer
    Converts numerical features to binary (0/1) features using a threashold
    #Example Result 
    #inputCol-feature, outputCol - binarized_feature , extra Param setThreshold(0.5)
    +---+-------+-----------------+
    |id |feature|binarized_feature|
    +---+-------+-----------------+
    |0  |0.1    |0.0              |
    |1  |0.8    |1.0              |
    |2  |0.2    |0.0              |
    +---+-------+-----------------+
    
◦PCA
    Converts a set of observations of possibly correlated variables into a set of values 
    of linearly uncorrelated variables called principal components
    #Example Result 
    #inputCol-features, outputCol - pcaFeatures , extra param setK(3)  
    +---------------------+-----------------------------------------------------------+
    |features             |pcaFeatures                                                |
    +---------------------+-----------------------------------------------------------+
    |(5,[1,3],[1.0,7.0])  |[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
    |[2.0,0.0,3.0,4.0,5.0]|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
    |[4.0,0.0,0.0,6.0,7.0]|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
    +---------------------+-----------------------------------------------------------+
    
◦PolynomialExpansion
    expands your features into a polynomial space, 
    which is formulated by an n-degree combination of original dimensions.
    #Example Result 
    #inputCol-features, outputCol - polyFeatures , extra param setDegree(3)
    +----------+------------------------------------------+
    |features  |polyFeatures                              |
    +----------+------------------------------------------+
    |[2.0,1.0] |[2.0,4.0,8.0,1.0,2.0,4.0,1.0,2.0,1.0]     |
    |[0.0,0.0] |[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]     |
    |[3.0,-1.0]|[3.0,9.0,27.0,-1.0,-3.0,-9.0,1.0,3.0,-1.0]|
    +----------+------------------------------------------+
    
◦Discrete Cosine Transform (DCT)
    transforms a length N real-valued Vector Column in the time domain 
    into another length N real-valued Vector COlumn  in the frequency domain
    #Example Result 
    #inputCol-features, outputCol - featuresDCT
    +--------------------+----------------------------------------------------------------+
    |features            |featuresDCT                                                     |
    +--------------------+----------------------------------------------------------------+
    |[0.0,1.0,-2.0,3.0]  |[1.0,-1.1480502970952693,2.0000000000000004,-2.7716385975338604]|
    |[-1.0,2.0,4.0,-7.0] |[-1.0,3.378492794482933,-7.000000000000001,2.9301512653149677]  |
    |[14.0,-2.0,-5.0,1.0]|[4.0,9.304453421915744,11.000000000000002,1.5579302036357163]   |
    +--------------------+----------------------------------------------------------------+
 
    
◦StringIndexer
     Encodes a string column of labels to a column of label indices. 
     The indices are in [0, numLabels), ordered by label frequencies, so the most frequent label gets index 0
    #Example Result 
    #inputCol-category, outputCol - categoryIndex
    +---+--------+-------------+
    |id |category|categoryIndex|
    +---+--------+-------------+
    |0  |a       |0.0          |
    |1  |b       |2.0          |
    |2  |c       |1.0          |
    |3  |a       |0.0          |
    |4  |a       |0.0          |
    |5  |c       |1.0          |
    +---+--------+-------------+
   
◦IndexToString
     Converts column of label indices back to a column containing the original labels as strings
    #Example Result 
    #inputCol-categoryIndex, outputCol - originalCategory
    +---+--------+-------------+----------------+
    |id |category|categoryIndex|originalCategory|
    +---+--------+-------------+----------------+
    |0  |a       |0.0          |a               |
    |1  |b       |2.0          |b               |
    |2  |c       |1.0          |c               |
    |3  |a       |0.0          |a               |
    |4  |a       |0.0          |a               |
    |5  |c       |1.0          |c               |
    +---+--------+-------------+----------------+
 
 
◦OneHotEncoder
    Maps a column of label indices to a column of binary vectors, with at most a single 1 value
    This encoding allows algorithms which expect continuous features
    such as Logistic Regression, to use categorical features
    eg Use StringIndexer and the OneHotEncoder to convert string category to column to be used for Logit 
    #Example Result 
    #inputCol-categoryIndex, outputCol - categoryVec
    +---+--------+-------------+-------------+
    |id |category|categoryIndex|categoryVec  |
    +---+--------+-------------+-------------+
    |0  |a       |0.0          |(2,[0],[1.0])|
    |1  |b       |2.0          |(2,[],[])    |
    |2  |c       |1.0          |(2,[1],[1.0])|
    |3  |a       |0.0          |(2,[0],[1.0])|
    |4  |a       |0.0          |(2,[0],[1.0])|
    |5  |c       |1.0          |(2,[1],[1.0])|
    +---+--------+-------------+-------------+
    
◦VectorIndexer
    Takes an input columns of type Vector and a parameter maxCategories 
    for max categories for encoding 
    Computes 0-based category indices for feature.
    #Example Result 
    #inputCol-features, outputCol - indexed , extra param setMaxCategories(2)
    +---+--------------+--------------+
    |id |features      |indexed       |
    +---+--------------+--------------+
    |0  |[1.0,0.5,-1.0]|[0.0,0.0,-1.0]|
    |1  |[1.0,1.0,1.0] |[0.0,1.0,1.0] |
    |2  |[4.0,0.5,2.0] |[1.0,0.0,2.0] |
    +---+--------------+--------------+
    
    
◦Interaction #(py not available)
    Takes vector or double-valued columns, and generates a single vector column 
    that contains the product of all combinations of one value from each input column
    #Example Result 
    #vec1<- VectorAssembler("id2", "id3", "id4"), 
    #vec2 <-VectorAssembler("id5", "id6", "id7")
    #inputCols-[vec1,vec2], outputCol - interactedCol
    +---+---+---+---+---+---+---+--------------+--------------+------------------------------------------------------+
    |id1|id2|id3|id4|id5|id6|id7|vec1          |vec2          |interactedCol                                         |
    +---+---+---+---+---+---+---+--------------+--------------+------------------------------------------------------+
    |1  |1  |2  |3  |8  |4  |5  |[1.0,2.0,3.0] |[8.0,4.0,5.0] |[8.0,4.0,5.0,16.0,8.0,10.0,24.0,12.0,15.0]            |
    |2  |4  |3  |8  |7  |9  |8  |[4.0,3.0,8.0] |[7.0,9.0,8.0] |[56.0,72.0,64.0,42.0,54.0,48.0,112.0,144.0,128.0]     |
    |3  |6  |1  |9  |2  |3  |6  |[6.0,1.0,9.0] |[2.0,3.0,6.0] |[36.0,54.0,108.0,6.0,9.0,18.0,54.0,81.0,162.0]        |
    |4  |10 |8  |6  |9  |4  |5  |[10.0,8.0,6.0]|[9.0,4.0,5.0] |[360.0,160.0,200.0,288.0,128.0,160.0,216.0,96.0,120.0]|
    |5  |9  |2  |7  |10 |7  |3  |[9.0,2.0,7.0] |[10.0,7.0,3.0]|[450.0,315.0,135.0,100.0,70.0,30.0,350.0,245.0,105.0] |
    |6  |1  |1  |4  |2  |8  |4  |[1.0,1.0,4.0] |[2.0,8.0,4.0] |[12.0,48.0,24.0,12.0,48.0,24.0,48.0,192.0,96.0]       |
    +---+---+---+---+---+---+---+--------------+--------------+------------------------------------------------------+

◦Normalizer
    transforms a dataset of Vector rows, normalizing each Vector to have p-norm
    #Example Result 
    #inputCol-features, outputCol - normFeatures , extra param setP(1.0)
    +---+--------------+------------------+
    |id |features      |normFeatures      |
    +---+--------------+------------------+
    |0  |[1.0,0.5,-1.0]|[0.4,0.2,-0.4]    |
    |1  |[2.0,1.0,1.0] |[0.5,0.25,0.25]   |
    |2  |[4.0,10.0,2.0]|[0.25,0.625,0.125]|
    +---+--------------+------------------+
 
◦StandardScaler
    transforms a dataset of rows, normalizing each feature(ie Column) 
    to have unit standard deviation and/or zero mean. 
    #Example Result 
    #inputCol-features, outputCol - normFeatures , extra param setWithStd(True),setWithMean(False)
    +---+--------------+------------------------------------------------------------+
    |id |features      |scaledFeatures                                              |
    +---+--------------+------------------------------------------------------------+
    |0  |[1.0,0.5,-1.0]|[0.6546536707079771,0.09352195295828246,-0.6546536707079771]|
    |1  |[2.0,1.0,1.0] |[1.3093073414159542,0.18704390591656492,0.6546536707079771] |
    |2  |[4.0,10.0,2.0]|[2.6186146828319083,1.8704390591656492,1.3093073414159542]  |
    +---+--------------+------------------------------------------------------------+

    
◦MinMaxScaler
    MinMaxScaler transforms a dataset of rows, 
    rescaling each feature(ie Column) to a specific range (often [0, 1]). 
    #Example Result 
    #inputCol-features, outputCol - scaledFeatures 
    +---+--------------+--------------+
    |id |features      |scaledFeatures|
    +---+--------------+--------------+
    |0  |[1.0,0.1,-1.0]|[0.0,0.0,0.0] |
    |1  |[2.0,1.1,1.0] |[0.5,0.1,0.5] |
    |2  |[3.0,10.1,3.0]|[1.0,1.0,1.0] |
    +---+--------------+--------------+

 
◦MaxAbsScaler
    transforms a dataset of rows, rescaling each feature(ie Column) to range [-1, 1]
    #Example Result 
    #inputCol-features, outputCol - scaledFeatures
    +---+--------------+----------------+
    |id |features      |scaledFeatures  |
    +---+--------------+----------------+
    |0  |[1.0,0.1,-8.0]|[0.25,0.01,-1.0]|
    |1  |[2.0,1.0,-4.0]|[0.5,0.1,-0.5]  |
    |2  |[4.0,10.0,8.0]|[1.0,1.0,1.0]   |
    +---+--------------+----------------+
   
◦Bucketizer
    transforms a column of continuous feature to a column of 
    feature bucket index after n splits
    #Example Result 
    #inputCol-features, outputCol - bucketedFeatures , with .setSplits(Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity))
    +--------+----------------+
    |features|bucketedFeatures|
    +--------+----------------+
    |-999.9  |0.0             |
    |-0.5    |1.0             |
    |-0.3    |1.0             |
    |0.0     |2.0             |
    |0.2     |2.0             |
    |999.9   |3.0             |
    +--------+----------------+

 
◦ElementwiseProduct
    scales each column/feature  of the dataset by a scalar multiplier.
    #Example Result 
    #inputCol-features, outputCol - transformedVector , with Vectors.dense(0.0, 1.0, 2.0)
    +---+-------------+-----------------+
    |id |vector       |transformedVector|
    +---+-------------+-----------------+
    |a  |[1.0,2.0,3.0]|[0.0,2.0,6.0]    |
    |b  |[4.0,5.0,6.0]|[0.0,5.0,12.0]   |
    +---+-------------+-----------------+
    
◦SQLTransformer
    transforms column via SQL statement 
    #Example Result "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 "
    +---+---+---+---+----+
    |id |v1 |v2 |v3 |v4  |
    +---+---+---+---+----+
    |0  |1.0|3.0|4.0|3.0 |
    |2  |2.0|5.0|7.0|10.0|
    +---+---+---+---+----+
    
◦VectorAssembler
    combines a given list of columns into a single vector column
    #Example Result 
    #inputCols-["hour", "mobile", "userFeatures"]  , outputCol - features ,    
    +---+----+------+--------------+-------+-----------------------+
    |id |hour|mobile|userFeatures  |clicked|features               |
    +---+----+------+--------------+-------+-----------------------+
    |0  |18  |1.0   |[0.0,10.0,0.5]|1.0    |[18.0,1.0,0.0,10.0,0.5]|
    +---+----+------+--------------+-------+-----------------------+
    
◦QuantileDiscretizer
    takes a column with continuous feature 
    and outputs a column of indexes to binned categorical feature( eg n bin)
    #Example Result 
    #inputCol-hour, outputCol - result , with extra Param setNumBuckets(3)
    +---+----+------+
    |id |hour|result|
    +---+----+------+
    |0  |18.0|2.0   |
    |1  |19.0|2.0   |
    |2  |8.0 |1.0   |
    |3  |5.0 |1.0   |
    |4  |2.2 |0.0   |
    +---+----+------+

    
##Spark - ML - QuickSummary - Feature Selectors , use with pyspark.ml.feature 
◦VectorSlicer
    extracts subfeatures from a vector column.
    #Example Result 
    #inputCol-userFeatures, outputCol - features , with extra Param setIndices([1]) or setIndices([1,2])
    +--------------------+-------------+
    |userFeatures        |features     |
    +--------------------+-------------+
    |(3,[0,1],[-2.0,2.3])|(2,[0],[2.3])|
    |[-2.0,2.3,0.0]      |[2.3,0.0]    |
    +--------------------+-------------+
    
◦RFormula
    selects columns specified by an R model formula
    #Example Result 
    #.setFormula("clicked ~ country + hour").setFeaturesCol("features").setLabelCol("label")
    # hence label from 'clicked' , features from 'country' and 'hour'
    +---+-------+----+-------+--------------+-----+
    |id |country|hour|clicked|features      |label|
    +---+-------+----+-------+--------------+-----+
    |7  |US     |18  |1.0    |[0.0,0.0,18.0]|1.0  |
    |8  |CA     |12  |0.0    |[1.0,0.0,12.0]|0.0  |
    |9  |NZ     |15  |0.0    |[0.0,1.0,15.0]|0.0  |
    +---+-------+----+-------+--------------+-----+

    
◦ChiSqSelector
    uses the Chi-Squared test of independence to decide 
    which features to choose(by numTopFeatures, percentile , fpr)
    #Example Result - top 1 features selected,
    #inputCol-features, outputCol - selectedFeatures , with extra param .setNumTopFeatures(1)
    +---+------------------+-------+----------------+
    |id |features          |clicked|selectedFeatures|
    +---+------------------+-------+----------------+
    |7  |[0.0,0.0,18.0,1.0]|1.0    |[18.0]          |
    |8  |[0.0,1.0,12.0,0.0]|0.0    |[12.0]          |
    |9  |[1.0,0.0,15.0,0.1]|0.0    |[15.0]          |
    +---+------------------+-------+----------------+
    
◦Userdefined Transformer 
    eg Add a constant term to input via extending UnaryTransformer
    #Example Result 
    #inputCol-input, outputCol - output , with extra Param .setShift(0.5)
    +-----+------+
    |input|output|
    +-----+------+
    |0.0  |0.5   |
    |1.0  |1.5   |
    |2.0  |2.5   |
    |3.0  |3.5   |
    |4.0  |4.5   |
    +-----+------+
    
◦Locality Sensitive Hashing
    A class of hashing techniques
    Two algorithms 
        Bucketed Random Projection for Euclidean Distance(hash bucket based on Euclidian distance)
        MinHash for Jaccard Distance 
    Various operations possibles
    #Example - Bucketed Random Projection for Euclidean Distance
    #inputCol-keys, outputCol - values , with extra Param .setBucketLength(2.0).setNumHashTables(3)
    +---+-----------+-----------------------+
    |id |keys       |values                 |
    +---+-----------+-----------------------+
    |0  |[1.0,1.0]  |[[0.0], [0.0], [-1.0]] |
    |1  |[1.0,-1.0] |[[-1.0], [-1.0], [0.0]]|
    |2  |[-1.0,-1.0]|[[-1.0], [-1.0], [0.0]]|
    |3  |[-1.0,1.0] |[[0.0], [0.0], [-1.0]] |
    +---+-----------+-----------------------+

    #Example - Approximate similarity join with threashold= 1.5 
    #inputCol = datasetA,datasetB (from above), outputCol= below full DF 
    #check each row, eg id 1 is joined with id 4 with distCol = 1.0 (min)
    +---------------------------------------------------+--------------------------------------------------+-------+
    |datasetA                                           |datasetB                                          |distCol|
    +---------------------------------------------------+--------------------------------------------------+-------+
    |[1,[1.0,-1.0],WrappedArray([-1.0], [-1.0], [0.0])] |[4,[1.0,0.0],WrappedArray([0.0], [-1.0], [-1.0])] |1.0    |
    |[0,[1.0,1.0],WrappedArray([0.0], [0.0], [-1.0])]   |[6,[0.0,1.0],WrappedArray([0.0], [0.0], [-1.0])]  |1.0    |
    |[1,[1.0,-1.0],WrappedArray([-1.0], [-1.0], [0.0])] |[7,[0.0,-1.0],WrappedArray([-1.0], [-1.0], [0.0])]|1.0    |
    |[3,[-1.0,1.0],WrappedArray([0.0], [0.0], [-1.0])]  |[5,[-1.0,0.0],WrappedArray([-1.0], [0.0], [0.0])] |1.0    |
    |[0,[1.0,1.0],WrappedArray([0.0], [0.0], [-1.0])]   |[4,[1.0,0.0],WrappedArray([0.0], [-1.0], [-1.0])] |1.0    |
    |[3,[-1.0,1.0],WrappedArray([0.0], [0.0], [-1.0])]  |[6,[0.0,1.0],WrappedArray([0.0], [0.0], [-1.0])]  |1.0    |
    |[2,[-1.0,-1.0],WrappedArray([-1.0], [-1.0], [0.0])]|[7,[0.0,-1.0],WrappedArray([-1.0], [-1.0], [0.0])]|1.0    |
    |[2,[-1.0,-1.0],WrappedArray([-1.0], [-1.0], [0.0])]|[5,[-1.0,0.0],WrappedArray([-1.0], [0.0], [0.0])] |1.0    |
    +---------------------------------------------------+--------------------------------------------------+-------+
    #Example  - Approximate nearest neighbor search, for Vectors.dense(1.0, 0.0)
    #inputCol = datsetA(from above), outputCol= below DF showing which ids are nearer via distCol=1.0 min 
    +---+----------+-----------------------+-------+
    |id |keys      |values                 |distCol|
    +---+----------+-----------------------+-------+
    |1  |[1.0,-1.0]|[[-1.0], [-1.0], [0.0]]|1.0    |
    |0  |[1.0,1.0] |[[0.0], [0.0], [-1.0]] |1.0    |
    +---+----------+-----------------------+-------+
    
    
    
    



###Spark - ML - Feature Extractors, Selection and transformation - Examples 
#Basic steps are 
# Create a Dataframe with inputCol and outputCol columns name , say x,y
# Instantiate CLASS with CLASS(inputCol='x', outCol='y')
#       For inputCols type, use multiple col as CLASS(inputCols=['x','x2'], outCol='y')
#       inputCol, outputCol can be set via .setInputCol('x') , .setOutputCol('y')
# For Transformer based , use instance.transform(inoutDF) to get transformed DataFrame
# For Exstimator based , use instance.fit(inputDF).transform(inoutDF)

#Note DataFrame may contain [ (DenseVector,) or (SparseVector,)] 
Vectors.sparse(numberRows, [(indices), (values)])
Vectors.dense([v1,v2,...]),)


##Spark - ML - Feature Extractors-  Example - TF-IDF 
#TF: Both HashingTF and CountVectorizer can be used to generate the term frequency vectors
#IDF: IDF is an Estimator which is fit on a dataset and produces an IDFModel. 
#it down-weights columns which appear frequently in a corpus.


from __future__ import print_function


from pyspark.ml.feature import HashingTF, IDF, Tokenizer

from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("TfIdfExample")\
        .getOrCreate()

    # $example on$
    sentenceData = spark.createDataFrame([
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
    ], ["label", "sentence"])

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.show()
    spark.stop()
#output 
+-----+--------------------+--------------------+--------------------+--------------------+
|label|            sentence|               words|         rawFeatures|  features          |
+-----+--------------------+--------------------+--------------------+--------------------+
|  0.0|Hi I heard about ...|[hi, i, heard, ab...|(20,[0,5,9,17],[1...|(20,[0,5,9,17],[0...|
|  0.0|I wish Java could...|[i, wish, java, c...|(20,[2,7,9,13,15]...|(20,[2,7,9,13,15]...|
|  1.0|Logistic regressi...|[logistic, regres...|(20,[4,6,13,15,18...|(20,[4,6,13,15,18...|
+-----+--------------------+--------------------+--------------------+--------------------+   
 

##Spark - ML - Feature Extractors- Example - Word2Vec
#maps each word to a unique fixed-size vector. 
#it takes extra Param
def  setMinCount(minCount: Int): 
def  setVectorSize(vectorSize: Int): 
def  setWindowSize(window: Int): 

#code - ml/word2vec_example.py
from pyspark.ml.feature import Word2Vec

# Input data: Each row is a bag of words from a sentence or document.
#Use Tokenizer in real code 
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)

result = model.transform(documentDF)
result.show()
#output 
+--------------------+--------------------+
|                text|              result|
+--------------------+--------------------+
|[Hi, I, heard, ab...|[0.01277976781129...|
|[I, wish, Java, c...|[0.07612769335641...|
|[Logistic, regres...|[-0.0675941422581...|
+--------------------+--------------------+



##Spark - ML - Feature Extractors- Example - CountVectorizer
#Convert text documents to vectors of token counts. 
#extra Param : http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.CountVectorizer
val  binary: BooleanParam 
val  minDF: DoubleParam 
val  minTF: DoubleParam
val  vocabSize: IntParam 
#Use set/get to update/access these


#code - ml/count_vectorizer_example.py

from pyspark.ml.feature import CountVectorizer

# Input data: Each row is a bag of words with a ID.
#Use Tokenizer in real code 
df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])

# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)

model = cv.fit(df)

result = model.transform(df)
result.show(truncate=False)
#output 
+---+---------------+-------------------------+
|id |words          |features                 |
+---+---------------+-------------------------+
|0  |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
|1  |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
+---+---------------+-------------------------+



##Spark - ML - Feature Transformer- Example -  StandardScalar 
#normalizing each feature(column) to have unit standard deviation and/or zero mean. 

#Extra Param :
•withStd: True by default. Scales the data to unit standard deviation.
•withMean: False by default. Centers the data with mean before scaling. It will build a dense output, so take care when applying to sparse input.

#Libsvm data file  
#data is stored in a sparse array/matrix form, each line is one instance (measurment)
#for regression, label is y value, for classification, label is target multi/label 
<label> <index1>:<value1> <index2>:<value2> ...
#index1 is vector index where value1 occurs (ie feature index)
#In short, +1 1:0.7 2:1 3:1 translates to:
#Assign to class +1, the point (0.7,1,1).



#code - ml/standard_scaler_example.py
from pyspark.ml.feature import StandardScaler

dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(dataFrame)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(dataFrame)
>>> scaledData.show()
+-----+--------------------+--------------------+
|label|            features|      scaledFeatures|
+-----+--------------------+--------------------+
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|









##Spark - ML - Feature Transformer- Example -  Normalizer 
#normalizing each Vector(sample) to have unit p-norm.,normed_data_p = power( Sum_of_pth_power_of_xi, 1/p) )
#Extra Param 
P 

#code - normalizer_example.py
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors

dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.5, -1.0]),),
    (1, Vectors.dense([2.0, 1.0, 1.0]),),
    (2, Vectors.dense([4.0, 10.0, 2.0]),)
], ["id", "features"])

# Normalize each Vector using $L^1$ norm.
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
l1NormData = normalizer.transform(dataFrame)
print("Normalized using L^1 norm")
l1NormData.show()
#output 
+---+--------------+------------------+
| id|      features|      normFeatures|
+---+--------------+------------------+
|  0|[1.0,0.5,-1.0]|    [0.4,0.2,-0.4]|
|  1| [2.0,1.0,1.0]|   [0.5,0.25,0.25]|
|  2|[4.0,10.0,2.0]|[0.25,0.625,0.125]|
+---+--------------+------------------+

# Normalize each Vector using $L^\infty$ norm.
lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
print("Normalized using L^inf norm")
lInfNormData.show()
#output
+---+--------------+--------------+
| id|      features|  normFeatures|
+---+--------------+--------------+
|  0|[1.0,0.5,-1.0]|[1.0,0.5,-1.0]|
|  1| [2.0,1.0,1.0]| [1.0,0.5,0.5]|
|  2|[4.0,10.0,2.0]| [0.4,1.0,0.2]|
+---+--------------+--------------+
















##Spark - ML - Feature Transformer- Example -   ElementwiseProduct 
#multiplies each input feature(column) by a provided 'weight' vector, element-wise

#code -  ml/elementwise_product_example.py
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors

# Create some vector data; also works for sparse vectors
data = [(Vectors.dense([1.0, 2.0, 3.0]),), (Vectors.dense([4.0, 5.0, 6.0]),)]
df = spark.createDataFrame(data, ["vector"])
transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]),
                                 inputCol="vector", outputCol="transformedVector")
# Batch transform the vectors to create new column:
transformer.transform(df).show()
#Output 
+-------------+-----------------+
|       vector|transformedVector|
+-------------+-----------------+
|[1.0,2.0,3.0]|    [0.0,2.0,6.0]|
|[4.0,5.0,6.0]|   [0.0,5.0,12.0]|
+-------------+-----------------+






##Spark - ML - Feature Transformer - Example -   PCA 
#to project vectors to a low-dimensional space 

#The example below shows how to project 5-dimensional feature vectors into 3-dimensional principal components.
#code -  ml/pca_example.py
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df)
result.show(truncate=False)
#output 
+---------------------+-----------------------------------------------------------+
|features             |pcaFeatures                                                |
+---------------------+-----------------------------------------------------------+
|(5,[1,3],[1.0,7.0])  |[1.6485728230883807,-4.013282700516296,-5.524543751369388] |
|[2.0,0.0,3.0,4.0,5.0]|[-4.645104331781534,-1.1167972663619026,-5.524543751369387]|
|[4.0,0.0,0.0,6.0,7.0]|[-6.428880535676489,-5.337951427775355,-5.524543751369389] |
+---------------------+-----------------------------------------------------------+













##Spark - ML - Feature Selection - Example -   ChiSqSelector 
#It operates on labeled data(labelCol='label') with categorical features(featuresCol='features'). 
#uses the Chi-Squared test of independence to decide which features to choose. 
#extra Param : It supports three selection methods: 
•numTopFeatures     chooses a fixed number of top features according to a chi-squared test. This is akin to yielding the features with the most predictive power.
•percentile         is similar to numTopFeatures but chooses a fraction of all features instead of a fixed number.
•fpr                chooses all features whose p-value is below a threshold, thus controlling the false positive rate of selection.


#code - ml/chisq_selector_example.py

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
    (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
    (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])

selector = ChiSqSelector(numTopFeatures=1, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="clicked")

result = selector.fit(df).transform(df)

result.show()
#output 
+---+------------------+-------+----------------+
| id|          features|clicked|selectedFeatures|
+---+------------------+-------+----------------+
|  7|[0.0,0.0,18.0,1.0]|    1.0|          [18.0]|
|  8|[0.0,1.0,12.0,0.0]|    0.0|          [12.0]|
|  9|[1.0,0.0,15.0,0.1]|    0.0|          [15.0]|
+---+------------------+-------+----------------+



##Spark - ML - Feature Transformer - Example -   Tokenization 
#Convert text (such as a sentence) to individual terms (usually words). 
#RegexTokenizer allows more advanced tokenization based on regular expression (regex) matching
#Extra Param
pattern:  split pattern in regex, default: "\\s+"


from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

sentenceDataFrame = spark.createDataFrame([
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["id", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
# alternatively: pattern="\\w+", gaps(False)

countTokens = udf(lambda words: len(words), IntegerType())

tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("sentence", "words").withColumn("tokens", countTokens(col("words"))).show(truncate=False)

regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.withColumn("tokens", countTokens(col("words"))).show(truncate=False)
#output 
+---+-----------------------------------+------------------------------------------+------+
|id |sentence                           |words                                     |tokens|
+---+-----------------------------------+------------------------------------------+------+
|0  |Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
|1  |I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
|2  |Logistic,regression,models,are,neat|[logistic, regression, models, are, neat] |5     |
+---+-----------------------------------+------------------------------------------+------+
    
    
    

##Spark - ML - Feature Transformer - Example -   StopWordsRemover
#Stop words are words which should be excluded from the input after Tokenizer 
#Extra Param , 
stopWords : Default available for few languages, 
            check by StopWordsRemover.loadDefaultStopWords(language)
            'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish' and 'turkish'.
caseSensitive : (default False)  

#code 
from pyspark.ml.feature import StopWordsRemover

sentenceData = spark.createDataFrame([
    (0, ["I", "saw", "the", "red", "balloon"]),
    (1, ["Mary", "had", "a", "little", "lamb"])
], ["id", "raw"])

remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
remover.transform(sentenceData).show(truncate=False)
#output 
+---+----------------------------+--------------------+
|id |raw                         |filtered            |
+---+----------------------------+--------------------+
|0  |[I, saw, the, red, balloon] |[saw, red, balloon] |
|1  |[Mary, had, a, little, lamb]|[Mary, little, lamb]|
+---+----------------------------+--------------------+





##Spark - ML - Feature Transformer - Example -   n-gram
#n-gram is represented by a space-delimited string of n  consecutive words
#Extra Param 
n   :Integer 

#code 
from pyspark.ml.feature import NGram

wordDataFrame = spark.createDataFrame([
    (0, ["Hi", "I", "heard", "about", "Spark"]),
    (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
    (2, ["Logistic", "regression", "models", "are", "neat"])
], ["id", "words"])

ngram = NGram(n=2, inputCol="words", outputCol="ngrams")

ngramDataFrame = ngram.transform(wordDataFrame)
ngramDataFrame.show(truncate=False)
#output 
+---+------------------------------------------+------------------------------------------------------------------+
|id |words                                     |ngrams                                                            |
+---+------------------------------------------+------------------------------------------------------------------+
|0  |[Hi, I, heard, about, Spark]              |[Hi I, I heard, heard about, about Spark]                         |
|1  |[I, wish, Java, could, use, case, classes]|[I wish, wish Java, Java could,could use, use case, case classes] |
|2  |[Logistic, regression, models, are, neat] |[Logistic regression, regression models, models are, are neat]    |
+---+------------------------------------------+------------------------------------------------------------------+




##Spark - ML - Feature Transformer - Example -   Binarizer
#Convert to 0/1 by thresholding numerical features 
#Extra Param 
threshold   : float 


from pyspark.ml.feature import Binarizer

continuousDataFrame = spark.createDataFrame([
    (0, 0.1),
    (1, 0.8),
    (2, 0.2)
], ["id", "feature"])

binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")

binarizedDataFrame = binarizer.transform(continuousDataFrame)
binarizedDataFrame.show()
#output 
+---+-------+-----------------+
| id|feature|binarized_feature|
+---+-------+-----------------+
|  0|    0.1|              0.0|
|  1|    0.8|              1.0|
|  2|    0.2|              0.0|
+---+-------+-----------------+



##Spark - ML - Feature Transformer - Example -  PolynomialExpansion
#expanding a features into n-degree combination of original dimensions. 
#Extra Param 
degree       : integer 


#The example below shows how to expand your features into a 3-degree polynomial space.
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (Vectors.dense([2.0, 1.0]),),
    (Vectors.dense([0.0, 0.0]),),
    (Vectors.dense([3.0, -1.0]),)
], ["features"])

polyExpansion = PolynomialExpansion(degree=3, inputCol="features", outputCol="polyFeatures")
polyDF = polyExpansion.transform(df)

polyDF.show(truncate=False)
#output 
+----------+------------------------------------------+
|features  |polyFeatures                              |
+----------+------------------------------------------+
|[2.0,1.0] |[2.0,4.0,8.0,1.0,2.0,4.0,1.0,2.0,1.0]     |
|[0.0,0.0] |[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]     |
|[3.0,-1.0]|[3.0,9.0,27.0,-1.0,-3.0,-9.0,1.0,3.0,-1.0]|
+----------+------------------------------------------+



##Spark - ML - Feature Transformer - Example -  Discrete Cosine Transform (DCT)?
#transforms a length N  real-valued sequence in the time domain into another length N 
#real-valued sequence in the frequency domain.

#Extra Param 
inverse     : boolean, True/False 

 

from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
    (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
    (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")
dctDf = dct.transform(df)
dctDf.show(truncate=False)
#output 
+--------------------+----------------------------------------------------------------+
|features            |featuresDCT                                                     |
+--------------------+----------------------------------------------------------------+
|[0.0,1.0,-2.0,3.0]  |[1.0,-1.1480502970952693,2.0000000000000004,-2.7716385975338604]|
|[-1.0,2.0,4.0,-7.0] |[-1.0,3.378492794482933,-7.000000000000001,2.9301512653149677]  |
|[14.0,-2.0,-5.0,1.0]|[4.0,9.304453421915744,11.000000000000002,1.5579302036357163]   |
+--------------------+----------------------------------------------------------------+



##Spark - ML - Feature Transformer - Example -  StringIndexer, IndexToString
StringIndexer 
    encodes a string feature/column to a column of indices. 
    The indices are in [0, numLabels), ordered by value frequencies, 
    so the most frequent label gets index 0.
IndexToString
    maps a column of indices back to a column containing the original strings
    inverse of StringIndexer

#code 
from pyspark.ml.feature import StringIndexer

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])

indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()
#output 
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          0.0|
|  1|       b|          2.0|
|  2|       c|          1.0|
|  3|       a|          0.0|
|  4|       a|          0.0|
|  5|       c|          1.0|
+---+--------+-------------+

converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
converted = converter.transform(indexed)
converted.show()
#output 
+---+--------+-------------+----------------+
| id|category|categoryIndex|originalCategory|
+---+--------+-------------+----------------+
|  0|       a|          0.0|               a|
|  1|       b|          2.0|               b|
|  2|       c|          1.0|               c|
|  3|       a|          0.0|               a|
|  4|       a|          0.0|               a|
|  5|       c|          1.0|               c|
+---+--------+-------------+----------------+




##Spark - ML - Feature Transformer - Example -   OneHotEncoder
#maps a column of indices(categorical eg from StringIndexer/VectorIndexer) to a column of binary vectors, 
#with at most a single one-value. 


from pyspark.ml.feature import OneHotEncoder, StringIndexer
df = spark.createDataFrame([
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")
], ["id", "category"])

stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()
#output 
+---+--------+-------------+-------------+
| id|category|categoryIndex|  categoryVec|
+---+--------+-------------+-------------+
|  0|       a|          0.0|(2,[0],[1.0])|
|  1|       b|          2.0|    (2,[],[])|
|  2|       c|          1.0|(2,[1],[1.0])|
|  3|       a|          0.0|(2,[0],[1.0])|
|  4|       a|          0.0|(2,[0],[1.0])|
|  5|       c|          1.0|(2,[1],[1.0])|
+---+--------+-------------+-------------+




##Spark - ML - Feature Transformer - Example -   VectorIndexer
#convert original values to category indices
# automatically decide which features are categorical in DataFrame 
1.Take an input column of type Vector and a parameter maxCategories.
2.Decide which features should be categorical based on the number of distinct values, 
  where features with at most maxCategories are declared categorical.
3.Compute 0-based category indices for each categorical feature.
4.Index categorical features and transform original feature values to indices.
#This transformed data could then be passed to algorithms such as DecisionTreeRegressor that handle categorical features.

#Extra Param 
maxCategories   : integer 


#code 
from pyspark.ml.feature import VectorIndexer

data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
indexerModel = indexer.fit(data)


# Create new column "indexed" with categorical values transformed to indices
indexedData = indexerModel.transform(data)
indexedData.show()
#output 
>>> indexedData.show()
+-----+--------------------+--------------------+
|label|            features|             indexed|
+-----+--------------------+--------------------+
|  0.0|(692,[127,128,129...|(692,[127,128,129...|
|  1.0|(692,[158,159,160...|(692,[158,159,160...|
|  1.0|(692,[124,125,126...|(692,[124,125,126...|
|  1.0|(692,[152,153,154...|(692,[152,153,154...|

>>> indexerModel.categoryMaps #index:{original value: index from VectorIndexer}
{222: {0.0: 0}, 592: {0.0: 0, 73.0: 1}, 100: {0.0: 0,88.0: 2, 165.0: 3, 11.0: 1, 222.0: 4}}






##Spark - ML - Feature Transformer - Example -   MinMaxScaler
#rescaling each feature to a specific range 
#Extra Param :
•min: 0.0 by default. Lower bound after transformation, shared by all features.
•max: 1.0 by default. Upper bound after transformation, shared by all features.


#code 
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors

dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -1.0]),),
    (1, Vectors.dense([2.0, 1.1, 1.0]),),
    (2, Vectors.dense([3.0, 10.1, 3.0]),)
], ["id", "features"])

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(dataFrame)

# rescale each feature to range [min, max].
scaledData = scalerModel.transform(dataFrame)
scaledData.show()
#output 
+---+--------------+--------------+
| id|      features|scaledFeatures|
+---+--------------+--------------+
|  0|[1.0,0.1,-1.0]| [0.0,0.0,0.0]|
|  1| [2.0,1.1,1.0]| [0.5,0.1,0.5]|
|  2|[3.0,10.1,3.0]| [1.0,1.0,1.0]|
+---+--------------+--------------+




##Spark - ML - Feature Transformer - Example -   MaxAbsScaler
#rescaling each feature to range [-1, 1]
#It does not shift/center the data, and thus does not destroy any sparsity.

#code 
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.linalg import Vectors

dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -8.0]),),
    (1, Vectors.dense([2.0, 1.0, -4.0]),),
    (2, Vectors.dense([4.0, 10.0, 8.0]),)
], ["id", "features"])

scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MaxAbsScalerModel
scalerModel = scaler.fit(dataFrame)
# rescale each feature to range [-1, 1].
scaledData = scalerModel.transform(dataFrame)
scaledData.show()
#output 
+---+--------------+----------------+
| id|      features|  scaledFeatures|
+---+--------------+----------------+
|  0|[1.0,0.1,-8.0]|[0.25,0.01,-1.0]|
|  1|[2.0,1.0,-4.0]|  [0.5,0.1,-0.5]|
|  2|[4.0,10.0,8.0]|   [1.0,1.0,1.0]|
+---+--------------+----------------+






##Spark - ML - Feature Transformer - Example -   Bucketizer
#transforms a column of continuous features to a column of feature bucket index 
#Extra Param 
splits: Parameter for mapping continuous features into buckets. 
        With n+1 splits, there are n buckets. 
        A bucket defined by splits x,y holds values in the range [x,y) except the last bucket

#code 
from pyspark.ml.feature import Bucketizer

splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]
data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]

dataFrame = spark.createDataFrame(data, ["features"])
bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")
# Transform original data into its bucket index.
bucketedData = bucketizer.transform(dataFrame)
bucketedData.show()
#output 
+--------+----------------+
|features|bucketedFeatures|
+--------+----------------+
|  -999.9|             0.0|
|    -0.5|             1.0|
|    -0.3|             1.0|
|     0.0|             2.0|
|     0.2|             2.0|
|   999.9|             3.0|
+--------+----------------+




##Spark - ML - Feature Transformer - Example -  SQLTransformer
#SQLTransformer implements the transformations which are defined by SQL statement
#SQLTransformer supports statements like:
•SELECT a, a + b AS a_b FROM __THIS__
•SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5
•SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b


#code 
from pyspark.ml.feature import SQLTransformer

df = spark.createDataFrame([
    (0, 1.0, 3.0),
    (2, 2.0, 5.0)
], ["id", "v1", "v2"])
sqlTrans = SQLTransformer(
    statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
sqlTrans.transform(df).show()
#output 
+---+---+---+---+----+
| id| v1| v2| v3|  v4|
+---+---+---+---+----+
|  0|1.0|3.0|4.0| 3.0|
|  2|2.0|5.0|7.0|10.0|
+---+---+---+---+----+


##Spark - ML - Feature Transformer - Example -  VectorAssembler
#combines a given list of columns(inputCols=['name,..]) into a single vector column. 
#Note in Spark, features is single column of list of all columns, not individual columns like in sklearn 
#Hence Convert individual columns into spark feature column 

#code 
from pyspark.sql.functions import * 
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.linalg import * 


dataset = spark.createDataFrame([
  (0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0),  
    ]).toDF("id", "hour", "mobile", "userFeatures", "clicked")

assembler = VectorAssembler().setInputCols(["hour", "mobile", "userFeatures"]).setOutputCol("features")

output = assembler.transform(dataset)
output.show()
#output 
+---+----+------+--------------+-------+--------------------+
| id|hour|mobile|  userFeatures|clicked|            features|
+---+----+------+--------------+-------+--------------------+
|  0|  18|   1.0|[0.0,10.0,0.5]|    1.0|[18.0,1.0,0.0,10....|
+---+----+------+--------------+-------+--------------------+




##Spark - ML - Feature Transformer - Example -  QuantileDiscretizer
#transforms column with continuous features to index of binned categorical features.
#Extra Param 
numBuckets : integer , The number of bins 
relativeError : float, When set to zero, exact quantiles are calculated


#code  
from pyspark.ml.feature import QuantileDiscretizer

data = [(0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2)]
df = spark.createDataFrame(data, ["id", "hour"])

discretizer = QuantileDiscretizer(numBuckets=3, inputCol="hour", outputCol="result")

result = discretizer.fit(df).transform(df)
result.show()
#outpput 
+---+----+------+
| id|hour|result|
+---+----+------+
|  0|18.0|   2.0|
|  1|19.0|   2.0|
|  2| 8.0|   1.0|
|  3| 5.0|   1.0|
|  4| 2.2|   0.0|
+---+----+------+




##Spark - ML - Feature Transformer - Example - VectorSlicer
#Converts a feature with list of values to new column with a sub-array of the original column. 
#It is useful for extracting features from a vector column.

#Extra Param: There are two types of indices,
1.Integer indices , setIndices([list_of_indices]).
2.String indices - names of features into the vector, setNames([list of names]).

#code 
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row

df = spark.createDataFrame([
    Row(userFeatures=Vectors.sparse(3, {0: -2.0, 1: 2.3})),
    Row(userFeatures=Vectors.dense([-2.0, 2.3, 0.0]))])

slicer = VectorSlicer(inputCol="userFeatures", outputCol="features", indices=[1])
output = slicer.transform(df)
output.show()
#output 
+--------------------+-------------+
|        userFeatures|     features|
+--------------------+-------------+
|(3,[0,1],[-2.0,2.3])|(1,[0],[2.3])|
|      [-2.0,2.3,0.0]|        [2.3]|
+--------------------+-------------+


##Spark - ML - Feature Transformer - Example - RFormula
#based on R Formula, create features(featuresCol) and label(labelCol) 
#Currently supports '~', '.', ':', '+', and '-'. 
#The basic operators are:
•~ separate target and terms
•+ concat terms, '+ 0' means removing intercept
•- remove a term, '- 1' means removing intercept
•: interaction (multiplication for numeric values, or binarized categorical values)
•. all columns except target

#Suppose a and b are double columns, 
•y ~ a + b means model y ~ w0 + w1 * a + w2 * b where w0 is the intercept and w1, w2 are coefficients.
•y ~ a + b + a:b - 1 means model y ~ w1 * a + w2 * b + w3 * a * b where w1, w2, w3 are coefficients.


#code 
from pyspark.ml.feature import RFormula
from pyspark.sql import SparkSession


dataset = spark.createDataFrame(
    [(7, "US", 18, 1.0),
     (8, "CA", 12, 0.0),
     (9, "NZ", 15, 0.0)],
    ["id", "country", "hour", "clicked"])

formula = RFormula(
    formula="clicked ~ country + hour",  #country is string categorical, hence would be one hot encoded 
    featuresCol="features",
    labelCol="label")  

output = formula.fit(dataset).transform(dataset)
output.show() #label = clicked, features = [country,hour]
#output 
+---+-------+----+-------+--------------+-----+
| id|country|hour|clicked|      features|label|
+---+-------+----+-------+--------------+-----+
|  7|     US|  18|    1.0|[0.0,0.0,18.0]|  1.0|
|  8|     CA|  12|    0.0|[0.0,1.0,12.0]|  0.0|
|  9|     NZ|  15|    0.0|[1.0,0.0,15.0]|  0.0|
+---+-------+----+-------+--------------+-----+









/******  Evaluator and Metrics  ******/
###Spark - ML - Evaluator 
class pyspark.ml.evaluation.BinaryClassificationEvaluator(*args, **kwargs)
    Evaluator for binary classification, 
    which expects two input columns: rawPrediction and label. 
    The rawPrediction column can be of type double (binary 0/1 prediction, or probability of label 1) 
    or of type vector (length-2 vector of raw predictions, scores, or label probabilities).
    copy(extra=None)
    evaluate(dataset, params=None)
        Evaluates the output with optional parameters.
        Parameters:
        •dataset – a dataset that contains labels/observations and predictions
        •params – an optional param map that overrides embedded params
        Returns:metric 
    explainParam(param)
    explainParams()
    extractParamMap(extra=None)
    getLabelCol()
    getMetricName()
    getOrDefault(param)
    getParam(paramName)
    getRawPredictionCol()
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isLargerBetter()
        Indicates whether the metric returned by evaluate() should be maximized 
        (True, default) or minimized (False). 
        A given evaluator may support multiple metrics which may be maximized or minimized.
    isSet(param)
        Checks whether a param is explicitly set by user.
    save(path)
    setLabelCol(value)
    setMetricName(value)
        Sets the value of metricName.
    setParams(self, rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
        Sets params for binary classification evaluator.
    setRawPredictionCol(value)
        Sets the value of rawPredictionCol.
    write()
        Returns an MLWriter instance for this ML instance.


#Example 
from pyspark.ml.linalg import Vectors
scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
   [(0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0)])
dataset = spark.createDataFrame(scoreAndLabels, ["raw", "label"])

evaluator = BinaryClassificationEvaluator(rawPredictionCol="raw")
>>> evaluator.metricName
Param(parent='BinaryClassificationEvaluator_422389751ad1f5450d99', name='metricN
ame', doc='metric name in evaluation (areaUnderROC|areaUnderPR)')

>>> evaluator.extractParamMap()
{Param(parent='BinaryClassificationEvaluator_422389751ad1f5450d99', name='metricName', doc='metric name in evaluation (areaUnderROC|areaUnderPR)'): 'areaUnderROC', 
Param(parent='BinaryClassificationEvaluator_422389751ad1f5450d99', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name.'): 'raw', 
Param(parent='BinaryClassificationEvaluator_422389751ad1f5450d99', name='labelCol', doc='label column name.'): 'label'}

>>> evaluator.evaluate(dataset)
0.70...
>>> evaluator.evaluate(dataset, {evaluator.metricName: "areaUnderPR"})
0.83...
>>> bce_path = temp_path + "/bce"
>>> evaluator.save(bce_path)
>>> evaluator2 = BinaryClassificationEvaluator.load(bce_path)
>>> str(evaluator2.getRawPredictionCol())
'raw'




class pyspark.ml.evaluation.RegressionEvaluator(*args, **kwargs)
    Evaluator for Regression, 
    which expects two input columns: prediction and label.
    copy(extra=None)
    evaluate(dataset, params=None)
        Evaluates the output with optional parameters.
        Parameters:
        •dataset – a dataset that contains labels/observations and predictions
        •params – an optional param map that overrides embedded params
        Returns:metric 
    explainParam(param)
    explainParams()
    extractParamMap(extra=None)
    getLabelCol()
    getMetricName()
    getOrDefault(param)
    getParam(paramName)
    getRawPredictionCol()
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isLargerBetter()
        Indicates whether the metric returned by evaluate() should be maximized 
        (True, default) or minimized (False). 
        A given evaluator may support multiple metrics which may be maximized or minimized.
    isSet(param)
        Checks whether a param is explicitly set by user.
    save(path)
    setLabelCol(value)
    setMetricName(value)
        Sets the value of metricName.
    setParams(self, predictionCol="prediction", labelCol="label", metricName="rmse")
        Sets params for binary classification evaluator.
    setRawPredictionCol(value)
        Sets the value of rawPredictionCol.
    write()
        Returns an MLWriter instance for this ML instance.
#Example 
scoreAndLabels = [(-28.98343821, -27.0), (20.21491975, 21.5),
    (-25.98418959, -22.0), (30.69731842, 33.0), (74.69283752, 71.0)]
dataset = spark.createDataFrame(scoreAndLabels, ["raw", "label"])
evaluator = RegressionEvaluator(predictionCol="raw")
>>> evaluator.metricName
Param(parent='RegressionEvaluator_468d9e41954f540d0624', name='metricName', doc='metric name in evaluation - one of:\n
                       rmse - root mean squared error (default)\n                       mse - mean squared error\n
                       r2 - r^2 metric\n                       
                       mae - mean absolute error.')
>>> evaluator.extractParamMap()
{Param(parent='RegressionEvaluator_468d9e41954f540d0624', name='predictionCol',doc='prediction column name.'): 'raw', 
Param(parent='RegressionEvaluator_468d9e41954f540d0624', name='metricName', doc='metric name in evaluation - one of:\n rmse - root mean squared error (default)\n  mse - mean squared error\n  r2 - r^2 metric\n  mae - mean absolute error.'): 'rmse', 
Param(parent='RegressionEvaluator_468d9e41954f540d0624', name='labelCol', doc='label column name.'): 'label'}
>>> evaluator.evaluate(dataset)
2.842...
>>> evaluator.evaluate(dataset, {evaluator.metricName: "r2"})
0.993...
>>> evaluator.evaluate(dataset, {evaluator.metricName: "mae"})
2.649...
>>> re_path = temp_path + "/re"
>>> evaluator.save(re_path)
>>> evaluator2 = RegressionEvaluator.load(re_path)
>>> str(evaluator2.getPredictionCol())
'raw'



class pyspark.ml.evaluation.MulticlassClassificationEvaluator(*args, **kwargs)
    Evaluator for Multiclass Classification, 
    which expects two input columns: prediction and label.
    copy(extra=None)
    evaluate(dataset, params=None)
        Evaluates the output with optional parameters.
        Parameters:
        •dataset – a dataset that contains labels/observations and predictions
        •params – an optional param map that overrides embedded params
        Returns:metric 
    explainParam(param)
    explainParams()
    extractParamMap(extra=None)
    getLabelCol()
    getMetricName()
    getOrDefault(param)
    getParam(paramName)
    getRawPredictionCol()
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isLargerBetter()
        Indicates whether the metric returned by evaluate() should be maximized 
        (True, default) or minimized (False). 
        A given evaluator may support multiple metrics which may be maximized or minimized.
    isSet(param)
        Checks whether a param is explicitly set by user.
    save(path)
    setLabelCol(value)
    setMetricName(value)
        Sets the value of metricName.
    setParams(self, predictionCol="prediction", labelCol="label", metricName="f1")
        Sets params for binary classification evaluator.
    setRawPredictionCol(value)
        Sets the value of rawPredictionCol.
    write()
        Returns an MLWriter instance for this ML instance.
#Example 
>>> scoreAndLabels = [(0.0, 0.0), (0.0, 1.0), (0.0, 0.0),
        (1.0, 0.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 0.0)]
>>> dataset = spark.createDataFrame(scoreAndLabels, ["prediction", "label"])
>>> evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
>>> evaluator.metricName
Param(parent='MulticlassClassificationEvaluator_4a1da6d0a3fe375accbf', 
name='metricName', 
doc='metric name in evaluation (f1|weightedPrecision|weightedRecall|accuracy)')

>>> evaluator.extractParamMap()
{Param(parent='MulticlassClassificationEvaluator_4a1da6d0a3fe375accbf', name='predictionCol', doc='prediction column name.'): 'prediction', 
Param(parent='MulticlassClassificationEvaluator_4a1da6d0a3fe375accbf', name='labelCol', doc='label column name.'): 'label', 
Param(parent='MulticlassClassificationEvaluator_4a1da6d0a3fe375accbf', name='metricName', doc='metric name in evaluation (f1|weightedPrecision|weightedRecall|accuracy)'): 'f1'}

>>> evaluator.evaluate(dataset)
0.66...
>>> evaluator.evaluate(dataset, {evaluator.metricName: "accuracy"})
0.66...
>>> mce_path = temp_path + "/mce"
>>> evaluator.save(mce_path)
>>> evaluator2 = MulticlassClassificationEvaluator.load(mce_path)
>>> str(evaluator2.getPredictionCol())
'prediction'




##confusion matrix used in finding accuracy in classification 
# T/F= True/False, P/N=Positive/Negative
#in best case , FP and FN  -> 0 
    True Positive (TP) - label is positive and prediction is also positive
    True Negative (TN) - label is negative and prediction is also negative
    False Positive (FP) - label is negative but prediction is positive
    False Negative (FN) - label is positive but prediction is negative

                actual +  actual - 
predicted +     TP          FP
predicted -     FN          TN

Recall = (TP)/(TP+FN)  -> 1(best)
Precision = TP/(TP+FP)  -> 1(best)


##for binary class evaluation of classification 
areaUnderROC  -> 1 is the best 

##A multiclass classification describes a classification problem 
#where there are M>2 possible labels for each data point 
#(the case where M=2 is the binary classification problem).

#For example, classifying handwriting samples to the digits 0 to 9, 
#having 10 possible classes. 
accuracy, f1 -> 1 is  the best 

Accuracy measures precision across all labels 
    - the number of times any class was predicted correctly (true positives) normalized by the number of data points. 
Precision by label considers only one class, 
    and measures the number of time a specific label was predicted correctly normalized by the number of times that label appears in the output.

#regression evaluation metrics 
r2  -> 1 is the best 
mean squared error -> 0 is the best 
pValues  -> 0 (<0.05) means coeffcients are significants 
aic   -> small  is the better model 



##Example of MulticlassClassificationEvaluator
data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]
# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]
# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
# train the model
model = trainer.fit(train)
# compute precision on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="precision")
print("Precision:" + str(evaluator.evaluate(predictionAndLabels)))



















/******  ML Pipeline ******/
##http://spark.apache.org/docs/latest/ml-pipeline.html
###Spark - ML - Pipeline 
##Steps 
1.Create all Estimators and transformers (extraction,selection,transformers, estimators) 
2.Create Pipleline via ctor arg, stages=[estimators])
3.Create pipline model, model = pipeline.fit(training)
  Calls sequentially transforms(training) anf fit(training) of all stages 
4.Create prediction, df = model.transform(test)

##Reference 
class pyspark.ml.Pipeline(*args, **kwargs)
    If stages is an empty list, the pipeline acts as an identity transformer.
    copy(extra=None)
    explainParam(param)
    explainParams()
    extractParamMap(extra=None)
    getOrDefault(param)
    getParam(paramName)
    getStages()
        Get pipeline stages.
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isSet(param)
    load(path)
    params
    classmethod read()
        Returns an MLReader instance for this class.
    save(path)
        Save this ML instance to the given path, a shortcut of write().save(path).
    setParams(self, stages=None)
        Sets params for Pipeline.
    setStages(value)
        Set pipeline stages.
        value – a list of transformers or estimators 
    write()
        Returns an MLWriter instance for this ML instance.
    fit(dataset, params=None)
        Fits a model to the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params. If a list/tuple of param maps is given, this calls fit on each param map and returns a list of models.
        Returns:fitted model(s)

class pyspark.ml.PipelineModel(stages)
    Represents a compiled pipeline with transformers and fitted models.
    copy(extra=None)
    explainParam(param)
    explainParams()
    extractParamMap(extra=None)
    getOrDefault(param)
    getParam(paramName)
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isSet(param)
    load(path)
    params
    classmethod read()
        Returns an MLReader instance for this class.
    save(path)
        Save this ML instance to the given path, a shortcut of write().save(path).
    setParams(self, stages=None)
        Sets params for Pipeline.
    write()
        Returns an MLWriter instance for this ML instance.
    transform(dataset, params=None)
        Transforms the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params.
 


##Example - LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001) #default featuresCol = 'features'
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))

>>> prediction.show()
+---+------------------+--------------------+--------------------+--------------------+--------------------+----------+
| id|              text|               words|            features|       rawPrediction|         probability|prediction|
+---+------------------+--------------------+--------------------+--------------------+--------------------+----------+
|  4|       spark i j k|    [spark, i, j, k]|(262144,[20197,24...|[-1.6609033227472...|[0.15964077387874...|       1.0|
|  5|             l m n|           [l, m, n]|(262144,[18910,10...|[1.64218895265644...|[0.83783256854767...|       0.0|
|  6|spark hadoop spark|[spark, hadoop, s...|(262144,[155117,2...|[-2.5980142174393...|[0.06926633132976...|       1.0|
|  7|     apache hadoop|    [apache, hadoop]|(262144,[66695,15...|[4.00817033336812...|[0.98215753334442...|       0.0|
+---+------------------+--------------------+--------------------+--------------------+--------------------+----------+



/****** ML: Tuning ******/
##https://spark.apache.org/docs/2.1.0/ml-tuning.html
###Spark - ML - Tuning - Model Selection 
#Tuning may be done for individual Estimators such as LogisticRegression 
#or Users can tune an entire Pipeline at once


#Supports CrossValidator and TrainValidationSplit. 
#These tools require the following items:
•Estimator
    An estimator or a Pipeline to tune
•Set of ParamMaps
    parameters to choose ,use ParamGridBuilder 
    Use estimator.PARAMETER and Array of values to choose from
    paramGrid = ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, [10, 100, 1000])
      .addGrid(lr.regParam, [0.1, 0.01])
      .build()                      
•Evaluator: 
    metric to measure how well a fitted Model does on held-out test data
        RegressionEvaluator for regression problems, 
        BinaryClassificationEvaluator for binary data, 
        MulticlassClassificationEvaluator for multiclass problems. 
    The default metric used to choose the best ParamMap 
    can be overridden by the setMetricName method in each of these evaluators.

•Get Best model via CrossValidator/TrainValidationSplit.best_model()
 after calling  instance.setEstimator(pipeline)
              .setEvaluator(evaluator_instance)
              .setEstimatorParamMaps(paramGrid)
              .setANY_OTHER_TUNING_PARAMETER

#Algorithm of the tool 
•They split the input data into separate training and test datasets.
•For each (training, test) pair, they iterate through the set of ParamMaps: 
   For each ParamMap, they fit the Estimator using those parameters, 
   get the fitted Model, and evaluate the Model's performance using the Evaluator.
•They select the Model produced by the best-performing set of parameters.


##Reference 
class pyspark.ml.tuning.ParamGridBuilder
Builder for a param grid used in grid search-based model selection.
    addGrid(param, values)
        Sets the given parameters in this grid to list of values 
    baseOn(*args)
        Sets the given parameters in this grid to fixed values. 
        Accepts either a parameter dictionary 
        or a list of (parameter, value) pairs.
    build()
        Builds and returns all combinations of parameters specified by the param grid.
        
#Example 
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression()
output = ParamGridBuilder() \
    .baseOn({lr.labelCol: 'l'}) \
    .baseOn([lr.predictionCol, 'p']) \
    .addGrid(lr.regParam, [1.0, 2.0]) \
    .addGrid(lr.maxIter, [1, 5]) \
    .build()
expected = [
    {lr.regParam: 1.0, lr.maxIter: 1, lr.labelCol: 'l', lr.predictionCol: 'p'},
    {lr.regParam: 2.0, lr.maxIter: 1, lr.labelCol: 'l', lr.predictionCol: 'p'},
    {lr.regParam: 1.0, lr.maxIter: 5, lr.labelCol: 'l', lr.predictionCol: 'p'},
    {lr.regParam: 2.0, lr.maxIter: 5, lr.labelCol: 'l', lr.predictionCol: 'p'}]
>>> len(output) == len(expected)
True
>>> all([m in expected for m in output])
True


class pyspark.ml.tuning.CrossValidator(*args, **kwargs)
    K-fold cross validation performs model selection by splitting the dataset into a set of non-overlapping randomly partitioned folds which are used as separate training and test datasets e.g., with k=3 folds, K-fold cross validation will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. Each fold is used as the test set exactly once.
    copy(extra=None)
    explainParams()
    extractParamMap(extra=None)
    getOrDefault(param)
    getParam(paramName)
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isSet(param)
    params
    getEstimator()
        Gets the value of estimator or its default value.
    getEstimatorParamMaps()
        Gets the value of estimatorParamMaps or its default value.
    getEvaluator()
        Gets the value of evaluator or its default value.
    getNumFolds()
        Gets the value of numFolds or its default value.
    getSeed()
        Gets the value of seed or its default value.
    setEstimator(value)
        Sets the value of estimator.
    setEstimatorParamMaps(value)
        Sets the value of estimatorParamMaps.
    setEvaluator(value)
        Sets the value of evaluator.
    setNumFolds(value)
        Sets the value of numFolds.
    setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3, seed=None): Sets params for cross validator.
    setSeed(value)
        Sets the value of seed.
    fit(dataset, params=None)
        Fits a model to the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params. If a list/tuple of param maps is given, this calls fit on each param map and returns a list of models.
        Returns:fitted model(s)
        
class pyspark.ml.tuning.CrossValidatorModel(bestModel, avgMetrics=[])
    CrossValidatorModel contains the model with the highest average cross-validation metric across folds and uses this model to transform input data. CrossValidatorModel also tracks the metrics for each param map evaluated.
    avgMetrics = None
        Average cross-validation metrics 
        for each paramMap in CrossValidator.estimatorParamMaps, 
        in the corresponding order.
    transform(dataset, params=None)
        Transforms the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params.
    Returns:    transformed dataset
    copy(extra=None)
    explainParams()
    extractParamMap(extra=None)
    getOrDefault(param)
    getParam(paramName)
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isSet(param)
    params
#Example 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
dataset = spark.createDataFrame(
    [(Vectors.dense([0.0]), 0.0),
     (Vectors.dense([0.4]), 1.0),
     (Vectors.dense([0.5]), 0.0),
     (Vectors.dense([0.6]), 1.0),
     (Vectors.dense([1.0]), 1.0)] * 10,
    ["features", "label"])
lr = LogisticRegression()
grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
cvModel = cv.fit(dataset)
>>> cvModel.avgMetrics[0]
0.5
>>> evaluator.evaluate(cvModel.transform(dataset))
0.8333...
>>> cv.extractParamMap()
{Param(parent='CrossValidator_4596b7b6b3c81cc38896', name='estimator', doc='estimator to be cross-validated'): LogisticRegression_4f819bc54f768933eb06, 
Param(parent='CrossValidator_4596b7b6b3c81cc38896', name='seed', doc='random seed.'): 7809051150349531440, 
Param(parent='CrossValidator_4596b7b6b3c81cc38896', name='evaluator', doc='evaluator used to select hyper-parameters that maximize the validator metric'): BinaryClassificationEvaluator_49f19113735956bd86ca, 
Param(parent='CrossValidator_4596b7b6b3c81cc38896', name='estimatorParamMaps', doc='estimatorparam maps')
    : [{Param(parent='LogisticRegression_4f819bc54f768933eb06', name='maxIter', doc='max number of iterations (>= 0).'): 0}, 
       {Param(parent='LogisticRegression_4f819bc54f768933eb06', name='maxIter', doc='max number of iterations (>=0).'): 1}], 
Param(parent='CrossValidator_4596b7b6b3c81cc38896', name='numFolds', doc='number of folds for cross validation'): 3}

##Another Example 
#with k=3 folds, CrossValidator will generate 3 (training, test) dataset pairs, 
#each of which uses 2/3 of the data for training and 1/3 for testing. 

#Note that cross-validation over a grid of parameters is expensive. 
#E.g., in the example below, the parameter grid has 3 values for hashingTF.numFeatures 
#and 2 values for lr.regParam, and CrossValidator uses 2 folds. 
#This multiplies out to (3×2)×2=12 different models being trained
 
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Prepare training documents, which are labeled.
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0),
    (4, "b spark who", 1.0),
    (5, "g d a y", 0.0),
    (6, "spark fly", 1.0),
    (7, "was mapreduce", 0.0),
    (8, "e spark program", 1.0),
    (9, "a e c l", 0.0),
    (10, "spark compile", 1.0),
    (11, "hadoop software", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(training)

# Prepare test documents, which are unlabeled.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "mapreduce spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents. cvModel uses the best model found (lrModel).
prediction = cvModel.transform(test)
>>> evaluator.evaluate(prediction)



 
class pyspark.ml.tuning.TrainValidationSplit(*args, **kwargs)
    Validation for hyper-parameter tuning. 
    Randomly splits the input dataset into train and validation sets, 
    and uses evaluation metric on the validation set to select the best model. 
    Similar to CrossValidator, but only splits the set once.
    copy(extra=None)
    explainParams()
    extractParamMap(extra=None)
    getOrDefault(param)
    getParam(paramName)
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isSet(param)
    params 
    getEstimator()
        Gets the value of estimator or its default value.
    getEstimatorParamMaps()
        Gets the value of estimatorParamMaps or its default value.
    getEvaluator()
        Gets the value of evaluator or its default value.
    getSeed()
        Gets the value of seed or its default value.
    getTrainRatio()
        Gets the value of trainRatio or its default value.
    setEstimator(value)
        Sets the value of estimator.
    setEstimatorParamMaps(value)
        Sets the value of estimatorParamMaps.
    setEvaluator(value)
        Sets the value of evaluator.
    setParams(self, estimator=None, estimatorParamMaps=None, evaluator=None, trainRatio=0.75, seed=None): 
        Sets params for the train validation split.
    setSeed(value)
        Sets the value of seed.
    setTrainRatio(value)
        Sets the value of trainRatio.
    fit(dataset, params=None)
        Fits a model to the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params. If a list/tuple of param maps is given, this calls fit on each param map and returns a list of models.
        Returns:    fitted model(s)

class pyspark.ml.tuning.TrainValidationSplitModel(bestModel, validationMetrics=[])
    Model from train validation split.
    bestModel = None
        best model from cross validation
    validationMetrics = None
        evaluated validation metrics
    transform(dataset, params=None)
        Transforms the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params.
        Returns: transformed dataset
    copy(extra=None)
    explainParams()
    extractParamMap(extra=None)
    getOrDefault(param)
    getParam(paramName)
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isSet(param)
    params  


#Example 

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
dataset = spark.createDataFrame(
    [(Vectors.dense([0.0]), 0.0),
     (Vectors.dense([0.4]), 1.0),
     (Vectors.dense([0.5]), 0.0),
     (Vectors.dense([0.6]), 1.0),
     (Vectors.dense([1.0]), 1.0)] * 10, ["features", "label"])
lr = LogisticRegression()
grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
evaluator = BinaryClassificationEvaluator()
tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
tvsModel = tvs.fit(dataset)
>>> evaluator.evaluate(tvsModel.transform(dataset))
0.8333...
>>> tvs.extractParamMap()
{Param(parent='TrainValidationSplit_4ac0bc35c0c9d884fbc6', name='trainRatio', doc='Param for ratio between train and     validation data. Must be between 0 and1.'): 0.75, 
Param(parent='TrainValidationSplit_4ac0bc35c0c9d884fbc6', name='estimator', doc='estimator to be cross-validated'): LogisticRegression_4e09821587057503cc01, 
Param(parent='TrainValidationSplit_4ac0bc35c0c9d884fbc6', name='estimatorParamMaps', doc='estimator param maps')
: [{Param(parent='LogisticRegression_4e09821587057503cc01', name='maxIter', doc='max number of iterations (>= 0).'): 0}
, {Param(parent='LogisticRegression_4e09821587057503cc01', name='maxIter', doc='max number of iterations (>= 0).'): 1}], 
Param(parent='TrainValidationSplit_4ac0bc35c0c9d884fbc6', name='seed', doc='random seed.'): -8807458389991577450, 
Param(parent='TrainValidationSplit_4ac0bc35c0c9d884fbc6', name='evaluator', doc='evaluator used to select hyper-parameters that maximize the validator metric'): BinaryClassificationEvaluator_4ee9bcf48318319f6feb}
>>> tvsModel.bestModel.summary.predictions.show()
+--------+-----+--------------------+--------------------+----------+
|features|label|       rawPrediction|         probability|prediction|
+--------+-----+--------------------+--------------------+----------+
|   [0.0]|  0.0|[-0.4054651081081...|[0.40000000000000...|       1.0|
|   [0.4]|  1.0|[-1.6333463351380...|[0.16337246239422...|       1.0|
|   [0.5]|  0.0|[-1.9403166418954...|[0.12561307421027...|       1.0|
|   [0.6]|  1.0|[-2.2472869486529...|[0.09558374418840...|       1.0|
|   [1.0]|  1.0|[-3.4751681756827...|[0.03002708980738...|       1.0|
|   [0.0]|  0.0|[-0.4054651081081...|[0.40000000000000...|       1.0|

>>> evaluator.evaluate(tvsModel.bestModel.summary.predictions)
0.8333333333333333
>>> tvsModel.validationMetrics
[0.5, 0.8095238095238095]


##Anotther example 
#TrainValidationSplit only evaluates each combination of parameters once, as opposed to k times in the case of CrossValidator. 
#It is therefore less expensive, but will not produce as reliable results when the training dataset is not sufficiently large.

#with trainRatio=0.75 
#TrainValidationSplit will generate a training and test dataset pair where 75% of the data is used for training and 25% for validation


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# Prepare training and test data.
data = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
train, test = data.randomSplit([0.9, 0.1], seed=12345)

lr = LinearRegression(maxIter=10)

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

# In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
model.transform(test).show()

    
    
    
    
    

/******  ML: Classifications - Logistic regression ******/
###Spark - ML - Classification, clustering and Regression - Quick Intro 
#Problem Type               Supported Methods
Binary Classification       linear SVMs, logistic regression, decision trees, random forests, gradient-boosted trees, naive Bayes 
Multiclass Classification   logistic regression, decision trees, random forests, naive Bayes 
Regression                  linear least squares, Lasso, ridge regression, decision trees, random forests, gradient-boosted trees, isotonic regression 

##General class structures 
CLASSIFICATION(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasMaxIter,
    HasRegParam, HasTol, HasRawPredictionCol, HasFitIntercept, HasStandardization,
    HasWeightCol, HasAggregationDepth, JavaMLWritable, JavaMLReadable)
    JavaEstimator = exposes .fit(df,params)
    HasFeaturesCol = exposes featuresCol (default 'features') and also setFeaturesCol(val)/getFeaturesCol()
    HasLabelCol = exposes labelCol (default 'label') and also setLabelCol(val)/getLabelCol()
    ...
    #JavaMLWritable, JavaMLReadable exposes 
    write()
    read()
    load(path)
        shortcut for read().load(path)
    save(path)
        shortcut for write().save(path)
    #from Params exposes following 
    copy(extra=None)
    explainParams()
    extractParamMap(extra=None)
    getOrDefault(param)
    getParam(paramName)
    hasDefault(param)
    hasParam(paramName)
    isDefined(param)
    isSet(param)
    params  
    
class Model(JavaModel, JavaClassificationModel, JavaMLWritable, JavaMLReadable)
    exposes 
        predictions = transform(df)
    coefficients()
    intercept()
    etc 
    #for Logistic,
    summary
        has predictions attributes etc 
        
#For evaluation Use 
MulticlassClassificationEvaluator(predictionCol="prediction")
    evaluate(predictions)
BinaryClassificationEvaluator(predictionCol="prediction")
    evaluate(predictions)
RegressionEvaluator(predictionCol="prediction")
    evaluate(predictions)
    
    
##List of classifications  - pyspark.ml.classification module
LinearSVC
    LinearSVCModel
LogisticRegression
    LogisticRegressionModel
    LogisticRegressionSummary
    LogisticRegressionTrainingSummary
    BinaryLogisticRegressionSummary
    BinaryLogisticRegressionTrainingSummary
DecisionTreeClassifier
    DecisionTreeClassificationModel
GBTClassifier
    GBTClassificationModel
RandomForestClassifier
    RandomForestClassificationModel
NaiveBayes
    NaiveBayesModel
MultilayerPerceptronClassifier
    MultilayerPerceptronClassificationModel
OneVsRest
    OneVsRestModel

##List of Regression  - pyspark.ml.regression module
AFTSurvivalRegression
    AFTSurvivalRegressionModel
DecisionTreeRegressor
    DecisionTreeRegressionModel
GBTRegressor
    GBTRegressionModel
GeneralizedLinearRegression
    GeneralizedLinearRegressionModel
    GeneralizedLinearRegressionSummary
    GeneralizedLinearRegressionTrainingSummary
IsotonicRegression
    IsotonicRegressionModel
LinearRegression
    LinearRegressionModel
    LinearRegressionSummary
    LinearRegressionTrainingSummary
RandomForestRegressor
    RandomForestRegressionModel

##List of Clustering - pyspark.ml.clustering module
BisectingKMeans
    BisectingKMeansModel
    BisectingKMeansSummary
KMeans
    KMeansModel
GaussianMixture
    GaussianMixtureModel
    GaussianMixtureSummary
LDA
    LDAModel
    LocalLDAModel
    DistributedLDAModel

##List of recomendation system - pyspark.ml.recommendation module
ALS
    ALSModel


    
    
    
###Spark - ML - Classification -  LogisticRegression (binary and multiclass classification )

#a binary label y is denoted as either +1  (positive) or -1  (negative), 
#which is convenient for the formulation. 
#However, the negative label is represented by 0  in spark instead of -1 , to be consistent with multiclass labeling.

class LogisticRegression(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasMaxIter,
                         HasRegParam, HasTol, HasProbabilityCol, HasRawPredictionCol,
                         HasElasticNetParam, HasFitIntercept, HasStandardization, HasThresholds,
                         HasWeightCol, HasAggregationDepth, JavaMLWritable, JavaMLReadable)
    setParams(featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, \
                  threshold=0.5, thresholds=None, probabilityCol="probability", \
                  rawPredictionCol="rawPrediction", standardization=True, weightCol=None, \
                  aggregationDepth=2, family="auto")
        Sets params for logistic regression.
        If the threshold and thresholds Params are both set, they must be equivalent.
                        
                         
#The purpose of the regularizer is to encourage simple models and avoid overfitting
#supported regularizer is L1, L2 and ElasticNet 
#L2-regularized problems are generally easier to solve than L1-regularized due to smoothness. 
#However, L1 regularization can help promote sparsity in weights leading to smaller and more interpretable models, the latter of which can be useful for feature selection. 
#Elastic net is a combination of L1 and L2 regularization. 
#It is not recommended to train models without any regularization, especially when the number of training examples is small.

##Binary Classification 
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
bdf = sc.parallelize([
    Row(label=1.0, weight=1.0, features=Vectors.dense(0.0, 5.0)),
    Row(label=0.0, weight=2.0, features=Vectors.dense(1.0, 2.0)),
    Row(label=1.0, weight=3.0, features=Vectors.dense(2.0, 1.0)),
    Row(label=0.0, weight=4.0, features=Vectors.dense(3.0, 3.0))]).toDF()
blor = LogisticRegression(regParam=0.01, weightCol="weight")
blorModel = blor.fit(bdf)
>>> blorModel.coefficients
DenseVector([-1.080..., -0.646...])
>>> blorModel.intercept
3.112...

test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, 1.0))]).toDF()
result = blorModel.transform(test0).head()
>>> result.prediction
1.0
>>> result.probability
DenseVector([0.02..., 0.97...])
>>> result.rawPrediction
DenseVector([-3.54..., 3.54...])

test1 = sc.parallelize([Row(features=Vectors.sparse(2, [0], [1.0]))]).toDF()
>>> blorModel.transform(test1).head().prediction
1.0

lr_path = temp_path + "/lr"
blor.save(lr_path)
lr2 = LogisticRegression.load(lr_path)
>>> lr2.getRegParam()
0.01

model_path = temp_path + "/lr_model"
blorModel.save(model_path)
model2 = LogisticRegressionModel.load(model_path)
>>> blorModel.coefficients[0] == model2.coefficients[0]
True
>>> blorModel.intercept == model2.intercept
True


##Example - Binary classification and Summary 
from __future__ import print_function

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("LogisticRegressionSummary") \
        .getOrCreate()

    # Load training data
    training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)

    # Extract the summary from the returned LogisticRegressionModel instance trained
    # in the earlier example
    trainingSummary = lrModel.summary

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)
    spark.stop()
    
    
    
##Multiclass
data_path = "data/mllib/sample_multiclass_classification_data.txt"
mdf = spark.read.format("libsvm").load(data_path)
mlor = LogisticRegression(regParam=0.1, elasticNetParam=1.0, family="multinomial")
mlorModel = mlor.fit(mdf)
>>> mlorModel.coefficientMatrix
SparseMatrix(3, 4, [0, 1, 2, 3], [3, 2, 1], [1.87..., -2.75..., -0.50...], 1)
>>> mlorModel.interceptVector
DenseVector([0.04..., -0.42..., 0.37...])





/******  ML: Classifications - Random Forest and Gradient Boost ******/

###Spark - ML - Classification & Regression - Decision trees and Tree Ensembles - ML and MLIB 

#Decision trees are widely used since they are easy to interpret, 
#handle categorical features, extend to the multiclass classification setting, 
#do not require feature scaling, 
#and are able to capture non-linearities and feature interactions. 


#The spark.ml implementation supports decision trees for binary and multiclass classification 
#and for regression, using both continuous and categorical features


##Spark - Ensemble Decision trees - Random Forests 
#Random forests combine many decision trees in order to reduce the risk of overfitting. 
#The spark.ml implementation supports random forests for binary and multiclass classification and for regression, using both continuous and categorical features.

class pyspark.ml.regression.RandomForestRegressor(*args, **kwargs)
    setParams(featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, \
                  maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, \
                  impurity="variance", subsamplingRate=1.0, seed=None, numTrees=20, \
                  featureSubsetStrategy="auto")
        Sets params for linear regression.
        
class pyspark.ml.classification.RandomForestClassifier(*args, **kwargs)
    setParams(featuresCol="features", labelCol="label", predictionCol="prediction", \
                 probabilityCol="probability", rawPredictionCol="rawPrediction", \
                  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, \
                  maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, seed=None, \
                  impurity="gini", numTrees=20, featureSubsetStrategy="auto", subsamplingRate=1.0)
        Sets params for linear classification.

#Code - Classsifiers 
"""
Random Forest Classifier Example.
"""
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifierExample")\
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =  VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only

    spark.stop()

#Code - regression 
"""
Random Forest Regressor Example.
"""
from __future__ import print_function


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestRegressorExample")\
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =  VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestRegressor(featuresCol="indexedFeatures")

    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, rf])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    rfModel = model.stages[1]
    print(rfModel)  # summary only

    spark.stop()

    
    
##Spark - Ensemble Decision trees - Gradient-Boosted Trees
#GBTs iteratively train decision trees in order to minimize a loss function. 
#The spark.ml implementation supports GBTs for binary classification 
#and for regression, using both continuous and categorical features.

class pyspark.ml.classification.GBTClassifier(*args, **kwargs)
    lossType = Param(parent='undefined', name='lossType', doc='Loss function which GBT tries to minimize (case-insensitive). Supported options: logistic')
    maxBins = Param(parent='undefined', name='maxBins', doc='Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature.')
    maxDepth = Param(parent='undefined', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.')¶
    setParams(featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, \
                  maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, \
                  lossType="logistic", maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0)
        Sets params for Gradient Boosted Tree Classification.

class pyspark.ml.regression.GBTRegressor(*args, **kwargs)
    lossType = Param(parent='undefined', name='lossType', doc='Loss function which GBT tries to minimize (case-insensitive). Supported options: squared, absolute')
    maxBins = Param(parent='undefined', name='maxBins', doc='Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature.')
    maxDepth = Param(parent='undefined', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.')
    setParams(featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, \
                  maxMemoryInMB=256, cacheNodeIds=False, subsamplingRate=1.0, \
                  checkpointInterval=10, lossType="squared", maxIter=20, stepSize=0.1, seed=None, \
                  impurity="variance")
        Sets params for Gradient Boosted Tree Regression.

#Code - classifier 
"""
Gradient Boosted Tree Classifier Example.
"""
from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("GradientBoostedTreeClassifierExample")\
        .getOrCreate()

    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =  VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    gbtModel = model.stages[2]
    print(gbtModel)  # summary only

    spark.stop()


##Code - regression 
"""
Gradient Boosted Tree Regressor Example.
"""
from __future__ import print_function

# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("GradientBoostedTreeRegressorExample")\
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a GBT model.
    gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)

    # Chain indexer and GBT in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, gbt])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    gbtModel = model.stages[1]
    print(gbtModel)  # summary only
    # $example off$

    spark.stop()
    
    
    
    
    
    
    
    
    
    
    
    
###Spark - classifications -  Naive Bayes 

#Naive Bayes is a simple multiclass classification algorithm 
#with the assumption of independence between every pair of features. 
#Naive Bayes can be trained very efficiently. 

#spark.mllib supports multinomial naive Bayes and Bernoulli naive Bayes. 

#These models are typically used for document classification. 
#Within that context, each observation is a document 
#and each feature represents a term whose value is the frequency of the term (in multinomial naive Bayes) 
#or a zero or one indicating whether the term was found in the document (in Bernoulli naive Bayes)

#modelType =  'multinomial' or 'bernoulli' with 'multinomial' as the default
#lambda = Additive smoothing  (default to 1.0 )

class pyspark.ml.classification.NaiveBayes(*args, **kwargs)[source]
    setParams(featuresCol="features", labelCol="label", predictionCol="prediction", \
                  probabilityCol="probability", rawPredictionCol="rawPrediction", smoothing=1.0, \
                  modelType="multinomial", thresholds=None, weightCol=None)
        Sets params for Naive Bayes.



#Code Example 
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load training data
data = spark.read.format("libsvm") \
    .load("data/mllib/sample_libsvm_data.txt")

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
model = nb.fit(train)

# select example rows to display.
predictions = model.transform(test)
predictions.show()

# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))






    
###Spark - Regression - LM and GLM - ML and MLIB 
#ordinary least squares or linear least squares uses no regularization; 
#ridge regression uses L2 regularization; 
#and Lasso uses L1 regularization
#L2 + L1 (elastic net) for ML 

##https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
LinearRegression
    elasticNetParam corresponds to L1/L2 ratio, alpha 
    regParam corresponds to regularization parameter (called lambda),  
        controls the overall strength of the penalty
    elasticParam in range [0, 1]. 
        For elasticParam = 0, the penalty is an L2 penalty(ridge). 
        For elasticParam = 1, it is an L1 penalty(lasso)
    the larger the value of lambda, alpha , 
        the greater the amount of shrinkage 
        and thus the coefficients become more robust to collinearity(features are corelated)
        but coefficients become sparse 
        
class pyspark.ml.regression.LinearRegression(*args, **kwargs)        
    setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction", 
                  maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True, 
                  standardization=True, solver="auto", weightCol=None, aggregationDepth=2)
        Sets params for linear regression    
        
        
GeneralizedLinearRegression
    #Family    Response Type        Links(* default/canonical)
    Gaussian    Continuous          Identity*, Log, Inverse 
    Binomial    Binary              Logit*, Probit, CLogLog 
    Poisson     Count               Log*, Identity, Sqrt 
    Gamma       Continuous          Inverse*, Idenity, Log 

class pyspark.ml.regression.GeneralizedLinearRegression(*args, **kwargs)
    family = Param(parent='undefined', name='family', doc='The name of family which is a description of the error distribution to be used in the model. Supported options: gaussian (default), binomial, poisson, gamma and tweedie.')
    link = Param(parent='undefined', name='link', doc='The name of link function which provides the relationship between the linear predictor and the mean of the distribution function. Supported options: identity, log, inverse, logit, probit, cloglog and sqrt.')
    linkPower = Param(parent='undefined', name='linkPower', doc='The index in the power link function. Only applicable to the Tweedie family.')
    linkPredictionCol = Param(parent='undefined', name='linkPredictionCol', doc='link prediction (linear predictor) column name')
    variancePower = Param(parent='undefined', name='variancePower', doc='The power in the variance function of the Tweedie distribution which characterizes the relationship between the variance and mean of the distribution. Only applicable for the Tweedie family. Supported values: 0 and [1, Inf).')
    setParams(self, labelCol="label", featuresCol="features", predictionCol="prediction", \
                  family="gaussian", link=None, fitIntercept=True, maxIter=25, tol=1e-6, \
                  regParam=0.0, weightCol=None, solver="irls", linkPredictionCol=None, \
                  variancePower=0.0, linkPower=None)
        Sets params for generalized linear regression.

    
    
#Example - LinearRegression
from pyspark.ml.regression import LinearRegression

# Load training data
training = spark.read.format("libsvm")\
    .load("data/mllib/sample_linear_regression_data.txt")

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


##training a GLM with a Gaussian response and identity link function and extracting model summary statistics.
from pyspark.ml.regression import GeneralizedLinearRegression

# Load training data
dataset = spark.read.format("libsvm")\
    .load("data/mllib/sample_linear_regression_data.txt")

glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

# Fit the model
model = glr.fit(dataset)

# Print the coefficients and intercept for generalized linear regression model
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Summarize the model over the training set and print out some metrics
summary = model.summary
print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()







###Spark - Regression - Survival regression - ML 
#Accelerated failure time (AFT) model 
#which is a parametric survival regression model for censored data. 
#It describes a model for the log of survival time
class pyspark.ml.regression.AFTSurvivalRegression(*args, **kwargs)
    censorCol = Param(parent='undefined', name='censorCol', doc='censor column name. The value of this column could be 0 or 1. If the value is 1, it means the event has occurred i.e. uncensored; otherwise censored.')
    getQuantileProbabilities()
        Gets the value of quantileProbabilities or its default value
    setParams(featuresCol="features", labelCol="label", predictionCol="prediction",
                  fitIntercept=True, maxIter=100, tol=1E-6, censorCol="censor",
                  quantileProbabilities=list([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
                  quantilesCol=None, aggregationDepth=2)
        Set all the params 
    Other Methods 

#Code - example 
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.linalg import Vectors
#censor = 1.0, event has occured, means censored , 
training = spark.createDataFrame([
    (1.218, 1.0, Vectors.dense(1.560, -0.605)),
    (2.949, 0.0, Vectors.dense(0.346, 2.158)),
    (3.627, 0.0, Vectors.dense(1.380, 0.231)),
    (0.273, 1.0, Vectors.dense(0.520, 1.151)),
    (4.199, 0.0, Vectors.dense(0.795, -0.226))], ["label", "censor", "features"])
quantileProbabilities = [0.3, 0.6]
aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                            quantilesCol="quantiles")

model = aft.fit(training)

# Print the coefficients, intercept and scale parameter for AFT survival regression
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))
print("Scale: " + str(model.scale))
model.transform(training).show(truncate=False)

##Another Example 
from pyspark.ml.linalg import Vectors
df = spark.createDataFrame([
    (1.0, Vectors.dense(1.0), 1.0),
    (1e-40, Vectors.sparse(1, [], []), 0.0)], ["label", "features", "censor"])
aftsr = AFTSurvivalRegression()
model = aftsr.fit(df)
>>> model.predict(Vectors.dense(6.3))
1.0
>>> model.predictQuantiles(Vectors.dense(6.3))
DenseVector([0.0101, 0.0513, 0.1054, 0.2877, 0.6931, 1.3863, 2.3026, 2.9957, 4.6052])
>>> model.transform(df).show()
+-------+---------+------+----------+
|  label| features|censor|prediction|
+-------+---------+------+----------+
|    1.0|    [1.0]|   1.0|       1.0|
|1.0E-40|(1,[],[])|   0.0|       1.0|
+-------+---------+------+----------+

aftsr_path = temp_path + "/aftsr"
aftsr.save(aftsr_path)
aftsr2 = AFTSurvivalRegression.load(aftsr_path)
>>> aftsr2.getMaxIter()
100




###Spark - Regression - IsotonicRegression
class pyspark.ml.regression.IsotonicRegression(*args, **kwargs)
    Currently implemented using parallelized pool adjacent violators algorithm. 
    Only univariate (single feature) algorithm supported
    featureIndex = Param(parent='undefined', name='featureIndex', doc='The index of the feature if featuresCol is a vector column, no effect otherwise.')
    isotonic = Param(parent='undefined', name='isotonic', doc='whether the output sequence should be isotonic/increasing (true) orantitonic/decreasing (false).')
    setParams(featuresCol="features", labelCol="label", 
                      predictionCol="prediction",
                      weightCol=None, isotonic=True, featureIndex=0)
        Set all the params 
    Other Methods    
#Example     
from pyspark.ml.linalg import Vectors
df = spark.createDataFrame([
    (1.0, Vectors.dense(1.0)),
    (0.0, Vectors.sparse(1, [], []))], ["label", "features"])
ir = IsotonicRegression()
model = ir.fit(df)
test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
>>> model.transform(test0).head().prediction
0.0
>>> model.boundaries
DenseVector([0.0, 1.0])

ir_path = temp_path + "/ir"
ir.save(ir_path)
ir2 = IsotonicRegression.load(ir_path)
>>> ir2.getIsotonic()
True





###Spark - Classification - Support Vector 
pyspark.ml.classification.LinearSVC(*args, **kwargs)
    setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxIter=100, regParam=0.0, tol=1e-6, rawPredictionCol="rawPrediction", \
                  fitIntercept=True, standardization=True, threshold=0.0, weightCol=None, \
                  aggregationDepth=2):
        Sets params for Linear SVM Classifier.
    Other methods 

from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
df = sc.parallelize([
    Row(label=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
    Row(label=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()
svm = LinearSVC(maxIter=5, regParam=0.01)
model = svm.fit(df)
>>> model.coefficients
DenseVector([0.0, -0.2792, -0.1833])
>>> model.intercept
1.0206118982229047
>>> model.numClasses
2
>>> model.numFeatures
3

test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, -1.0, -1.0))]).toDF()
result = model.transform(test0).head()
>>> result.prediction
1.0
>>> result.rawPrediction
DenseVector([-1.4831, 1.4831])

svm_path = temp_path + "/svm"
svm.save(svm_path)
svm2 = LinearSVC.load(svm_path)
>>> svm2.getMaxIter()
5

model_path = temp_path + "/svm_model"
model.save(model_path)
model2 = LinearSVCModel.load(model_path)
>>> model.coefficients[0] == model2.coefficients[0]
True
>>> model.intercept == model2.intercept
True


###Spark - Classification - MultilayerPerceptron

class pyspark.ml.classification.MultilayerPerceptronClassifier(*args, **kwargs)[source]
    Classifier trainer based on the Multilayer Perceptron. 
    Each layer has sigmoid activation function, output layer has softmax. 
    Number of inputs has to be equal to the size of feature vectors. 
    Number of outputs has to be equal to the total number of labels.
    setParams(self, featuresCol="features", labelCol="label", 
            predictionCol="prediction", maxIter=100, tol=1e-6, seed=None, 
            layers=None, blockSize=128, stepSize=0.03, solver="l-bfgs", 
            initialWeights=None)
        Sets params for MultilayerPerceptronClassifier.
    Other methods 
    
#Example 
from pyspark.ml.linalg import Vectors
df = spark.createDataFrame([
    (0.0, Vectors.dense([0.0, 0.0])),
    (1.0, Vectors.dense([0.0, 1.0])),
    (1.0, Vectors.dense([1.0, 0.0])),
    (0.0, Vectors.dense([1.0, 1.0]))], ["label", "features"])
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=[2, 2, 2], blockSize=1, seed=123)
model = mlp.fit(df)
>>> model.layers
[2, 2, 2]
>>> model.weights.size
12

testDF = spark.createDataFrame([
    (Vectors.dense([1.0, 0.0]),),
    (Vectors.dense([0.0, 0.0]),)], ["features"])
>>> model.transform(testDF).show()
+---------+----------+
| features|prediction|
+---------+----------+
|[1.0,0.0]|       1.0|
|[0.0,0.0]|       0.0|
+---------+----------+


mlp_path = temp_path + "/mlp"
mlp.save(mlp_path)
mlp2 = MultilayerPerceptronClassifier.load(mlp_path)
>>> mlp2.getBlockSize()
1

model_path = temp_path + "/mlp_model"
model.save(model_path)
model2 = MultilayerPerceptronClassificationModel.load(model_path)
>>> model.layers == model2.layers
True
>>> model.weights == model2.weights
True

mlp2 = mlp2.setInitialWeights(list(range(0, 12)))
model3 = mlp2.fit(df)
>>> model3.weights != model2.weights
True
>>> model3.layers == model.layers
True



###Spark - ML - recommendation
#Given user, item, ratings, predicts user's recommendations 

class pyspark.ml.recommendation.ALS(*args, **kwargs)[source]
    Alternating Least Squares (ALS) matrix factorization.
    coldStartStrategy = Param(parent='undefined', name='coldStartStrategy', doc="strategy for dealing with unknown or new users/items at prediction time. This may be useful in cross-validation or production scenarios, for handling user/item ids the model has not seen in the training data. Supported values: 'nan', 'drop'.")
    implicitPrefs = Param(parent='undefined', name='implicitPrefs', doc='whether to use implicit preference')
    nonnegative = Param(parent='undefined', name='nonnegative', doc='whether to use nonnegative constraint for least squares')numItemBlocks = Param(parent='undefined', name='numItemBlocks', doc='number of item blocks')
    numUserBlocks = Param(parent='undefined', name='numUserBlocks', doc='number of user blocks')
    rank = Param(parent='undefined', name='rank', doc='rank of the factorization')
    ratingCol = Param(parent='undefined', name='ratingCol', doc='column name for ratings')¶
    setParams(self, rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10, \
                 implicitPrefs=False, alpha=1.0, userCol="user", itemCol="item", seed=None, \
                 ratingCol="rating", nonnegative=False, checkpointInterval=10, \
                 intermediateStorageLevel="MEMORY_AND_DISK", \
                 finalStorageLevel="MEMORY_AND_DISK", coldStartStrategy="nan")
        Sets params for ALS.

class pyspark.ml.recommendation.ALSModel(java_model=None)
    rank
        rank of the matrix factorization model
    itemFactors
        a DataFrame that stores item factors in two columns: id and features
    recommendForAllItems(numUsers)
        Returns top numUsers users recommended for each item, for all items.
        numUsers – max number of recommendations for each item 
        Returns:a DataFrame of (itemCol, recommendations), where recommendations are stored as an array of (userCol, rating) Rows. 
    recommendForAllUsers(numItems)
        Returns top numItems items recommended for each user, for all users.
        numItems – max number of recommendations for each user 
        Returns:a DataFrame of (userCol, recommendations), where recommendations are stored as an array of (itemCol, rating) Rows. 
    transform(dataset, params=None)
        Transforms the input dataset with optional parameters.
        Parameters:
        •dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        •params – an optional param map that overrides embedded params.
        Returns:transformed dataset
    userFactors
        a DataFrame that stores user factors in two columns: id and features
    Other Methods 


#Example 
df = spark.createDataFrame(
    [(0, 0, 4.0), (0, 1, 2.0), (1, 1, 3.0), (1, 2, 4.0), (2, 1, 1.0), (2, 2, 5.0)],
    ["user", "item", "rating"])
als = ALS(rank=10, maxIter=5, seed=0)
model = als.fit(df)
>>> model.rank
10
>>> model.userFactors.orderBy("id").collect()
[Row(id=0, features=[...]), Row(id=1, ...), Row(id=2, ...)]

test = spark.createDataFrame([(0, 2), (1, 0), (2, 0)], ["user", "item"])
predictions = sorted(model.transform(test).collect(), key=lambda r: r[0])
>>> predictions[0]
Row(user=0, item=2, prediction=-0.13807615637779236)
>>> predictions[1]
Row(user=1, item=0, prediction=2.6258413791656494)
>>> predictions[2]
Row(user=2, item=0, prediction=-1.5018409490585327)

>>> user_recs = model.recommendForAllUsers(3)
>>> user_recs.where(user_recs.user == 0).select("recommendations.item", "recommendations.rating").collect()
[Row(item=[0, 1, 2], rating=[3.910..., 1.992..., -0.138...])]
>>> item_recs = model.recommendForAllItems(3)
>>> item_recs.where(item_recs.item == 2)        .select("recommendations.user", "recommendations.rating").collect()
[Row(user=[2, 1, 0], rating=[4.901..., 3.981..., -0.138...])]


als_path = temp_path + "/als"
als.save(als_path)
als2 = ALS.load(als_path)
>>> als.getMaxIter()
5






/******  ML: Clustering - KMeans  ******/
###Spark - ML - Clustering 
#Clustering is an unsupervised learning problem 
#whereby we aim to group subsets of entities with one another based on some notion of similarity

class pyspark.ml.clustering.BisectingKMeans(*args, **kwargs)
    minDivisibleClusterSize = Param(parent='undefined', name='minDivisibleClusterSize', doc='The minimum number of points (if >= 1.0) or the minimum proportion of points (if < 1.0) of a divisible cluster.')
    k = Param(parent='undefined', name='k', doc='The desired number of leaf clusters. Must be > 1.')
    setParams(featuresCol="features", predictionCol="prediction", maxIter=20, \
                      seed=None, k=4, minDivisibleClusterSize=1.0)
        Sets params for BisectingKMeans.


class pyspark.ml.clustering.KMeans(*args, **kwargs)
    K-means clustering with a k-means++ like initialization mode (the k-means|| algorithm by Bahmani et al).
    initMode = Param(parent='undefined', name='initMode', doc='The initialization algorithm. This can be either "random" to choose random points as initial cluster centers, or "k-means||" to use a parallel variant of k-means++')
    initSteps = Param(parent='undefined', name='initSteps', doc='The number of steps for k-means|| initialization mode. Must be > 0.')
    k = Param(parent='undefined', name='k', doc='The number of clusters to create. Must be > 1.')
    setParams(featuresCol="features", predictionCol="prediction", k=2, \
                      initMode="k-means||", initSteps=2, tol=1e-4, maxIter=20, seed=None)
        Sets params for KMeans.


KMeansModel
    clusterCenters()
        Get the cluster centers, represented as a list of NumPy arrays.
    computeCost(dataset)
        Return the K-means cost (sum of squared distances of points to their nearest center) for this model on the given data.
    summary
        Gets summary (e.g. cluster assignments, cluster sizes) of the model trained on the training set. An exception is thrown if no summary exists
    transform(dataset, params=None)
        Transforms the input dataset with optional parameters
    Other Methods 
        
##Code - Kmeans 
from pyspark.ml.clustering import KMeans

# Loads data.
dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

##Code - Bisecting k-means
#Strategies for hierarchical clustering generally fall into two types:
•Agglomerative: This is a 'bottom up' approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
•Divisive: This is a 'top down' approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

#ML  -Only top-down approach 
from pyspark.ml.clustering import BisectingKMeans

# Loads data.
dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(dataset)

# Evaluate clustering.
cost = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(cost))

# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)   
    
    
###Spark - Clustering - LDA 
#Latent Dirichlet Allocation (LDA), a topic model designed for text documents.
#Terminology:
•'term' = 'word': an el
•'token': instance of a term appearing in a document
•'topic': multinomial distribution over terms representing some concept
•'document': one piece of text, corresponding to one row in the input data
#Given list of documents, automatically discovering topics that these documents contain


class pyspark.ml.clustering.LDA(*args, **kwargs)
    Two types of Models:
        Local (non-distributed) model fitted by LDA. 
            This model stores the inferred topics only; it does not store info about the training dataset.
        Distributed model fitted by LDA. 
            This type of model is produced by Expectation-Maximization (EM).
            This model stores the inferred topics, the full training dataset, and the topic distribution for each training document
    k = Param(parent='undefined', name='k', doc='The number of topics (clusters) to infer. Must be > 1.')
    learningDecay = Param(parent='undefined', name='learningDecay', doc='Learning rate, set as anexponential decay rate. This should be between (0.5, 1.0] to guarantee asymptotic convergence.')
    learningOffset = Param(parent='undefined', name='learningOffset', doc='A (positive) learning parameter that downweights early iterations. Larger values make early iterations count less')
    optimizeDocConcentration = Param(parent='undefined', name='optimizeDocConcentration', doc='Indicates whether the docConcentration (Dirichlet parameter for document-topic distribution) will be optimized during training.')
    optimizer = Param(parent='undefined', name='optimizer', doc='Optimizer or inference algorithm used to estimate the LDA model. Supported: online, em')
    subsamplingRate = Param(parent='undefined', name='subsamplingRate', doc='Fraction of the corpus to be sampled and used in each iteration of mini-batch gradient descent, in range (0, 1].')
    topicConcentration = Param(parent='undefined', name='topicConcentration', doc='Concentration parameter (commonly named "beta" or "eta") for the prior placed on topic\' distributions over terms.')
    topicDistributionCol = Param(parent='undefined', name='topicDistributionCol', doc='Output column with estimates of the topic mixture distribution for each document (often called "theta" in the literature). Returns a vector of zeros for an empty document.')
    setParams(featuresCol="features", maxIter=20, seed=None, checkpointInterval=10,\
                  k=10, optimizer="online", learningOffset=1024.0, learningDecay=0.51,\
                  subsamplingRate=0.05, optimizeDocConcentration=True,\
                  docConcentration=None, topicConcentration=None,\
                  topicDistributionCol="topicDistribution", keepLastCheckpoint=True):

        Sets params for LDA.

class pyspark.ml.clustering.LDAModel(java_model=None)
    describeTopics(maxTermsPerTopic=10)
        Return the topics described by their top-weighted terms.
    estimatedDocConcentration()
        Value for LDA.docConcentration estimated from data. 
        If Online LDA was used and LDA.optimizeDocConcentration was set to false, 
        then this returns the fixed (given) value for the LDA.docConcentration parameter.
    isDistributed()
        Indicates whether this instance is of type DistributedLDAModel
    logLikelihood(dataset)
        Calculates a lower bound on the log likelihood of the entire corpus. 
        For DistributedModel, complexity is high 
    logPerplexity(dataset)
        Calculate an upper bound on perplexity. (Lower is better.) 
        For DistributedModel, complexity is high 
    topicsMatrix()
        Inferred topics, where each topic is represented by a distribution over terms. 
        This is a matrix of size vocabSize x k, where each column is a topic. 
        For DistributedModel, complexity is high 
    transform(dataset, params=None)
        Transforms the input dataset with optional parameters
    vocabSize()
        Vocabulary size (number of terms or words in the vocabulary)
    Other Methods 

#Example 
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA
df = spark.createDataFrame([[1, Vectors.dense([0.0, 1.0])],
     [2, SparseVector(2, {0: 1.0})],], ["id", "features"])
lda = LDA(k=2, seed=1, optimizer="em")
model = lda.fit(df)
>>> model.isDistributed()
True

>>> localModel = model.toLocal()
>>> localModel.isDistributed()
False
>>> model.vocabSize()
2
>>> model.describeTopics().show()
+-----+-----------+--------------------+
|topic|termIndices|         termWeights|
+-----+-----------+--------------------+
|    0|     [1, 0]|[0.50401530077160...|
|    1|     [0, 1]|[0.50401530077160...|
+-----+-----------+--------------------+
...
>>> model.topicsMatrix()
DenseMatrix(2, 2, [0.496, 0.504, 0.504, 0.496], 0)


lda_path = temp_path + "/lda"
lda.save(lda_path)
sameLDA = LDA.load(lda_path)
distributed_model_path = temp_path + "/lda_distributed_model"
model.save(distributed_model_path)
sameModel = DistributedLDAModel.load(distributed_model_path)
local_model_path = temp_path + "/lda_local_model"
localModel.save(local_model_path)
sameLocalModel = LocalLDAModel.load(local_model_path)

 
###Spark - Clustering - GaussianMixture clustering
#Given a set of sample points, this class will maximize the log-likelihood for a mixture of k Gaussians
#you can perform GMM when you know that the data points are mixtures of a gaussian distribution. 
#Basically forming clusters with different mean and standard deviation

class pyspark.ml.clustering.GaussianMixture(*args, **kwargs)
    k = Param(parent='undefined', name='k', doc='Number of independent Gaussians in the mixture model. Must be > 1.')
    setParams(featuresCol="features", predictionCol="prediction", k=2, \
                  probabilityCol="probability", tol=0.01, maxIter=100, seed=None)
        Sets params for GaussianMixture.
        
class pyspark.ml.clustering.GaussianMixtureModel(java_model=None)
    gaussiansDF
        Retrieve Gaussian distributions as a DataFrame. 
        Each row represents a Gaussian Distribution. The DataFrame has two columns: mean (Vector) and cov (Matrix).
    summary
        Gets summary (e.g. cluster assignments, cluster sizes) of the model trained on the training set. An exception is thrown if no summary exists.
    Other Methods 
    
    
#Code - Example 

from pyspark.ml.linalg import Vectors
data = [(Vectors.dense([-0.1, -0.05 ]),),
        (Vectors.dense([-0.01, -0.1]),),
        (Vectors.dense([0.9, 0.8]),),
        (Vectors.dense([0.75, 0.935]),),
        (Vectors.dense([-0.83, -0.68]),),
        (Vectors.dense([-0.91, -0.76]),)]
df = spark.createDataFrame(data, ["features"])
gm = GaussianMixture(k=3, tol=0.0001, maxIter=10, seed=10)
>>> model = gm.fit(df)
>>> model.hasSummary
True
>>> summary = model.summary
>>> summary.k
3
>>> summary.clusterSizes
[2, 2, 2]
>>> summary.logLikelihood
8.14636...
>>> weights = model.weights
>>> len(weights)
3
>>> model.gaussiansDF.select("mean").head()
Row(mean=DenseVector([0.825, 0.8675]))
>>> model.gaussiansDF.select("cov").head()
Row(cov=DenseMatrix(2, 2, [0.0056, -0.0051, -0.0051, 0.0046], False))

>>> transformed = model.transform(df).select("features", "prediction")
>>> rows = transformed.collect()
>>> rows[4].prediction == rows[5].prediction
True
>>> rows[2].prediction == rows[3].prediction
True

>>> gmm_path = temp_path + "/gmm"
>>> gm.save(gmm_path)
>>> gm2 = GaussianMixture.load(gmm_path)
>>> gm2.getK()
3

###Spark - Plotting Graphs - Plotly - Introductions 
#https://plot.ly/python/apache-spark/

##Installation 
$ pip install  plotly
$ pip install ta-lib*.whl #download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
$ pip install cufflinks
$ pip install jupyter --upgrade
$ pip install nbconvert --upgrade #some bug fix 
#https://plot.ly/python/


##Plotly - a JS framework (does not depend on matplotlib)
1. it has online mode thorugh which you can save image file 
   requires signup and getting api_key 
   plotly.iplot(in jupyter) / plotly.plot(commandline) would save the file to online server 
   under 'username'  and with file_id
2. It has offline mode, saving local file via 'plot' args image='png', 
   but need to download the actual png file by browser
   jupyter can save it to html/pdf, but 'Out' would not be displayed 
   For that, you need again online mode and embed the picture in notebook 
3. Note plotly.iplot for jupyter and plotly.plot is for command line 
   both display graph in an html(as plotly is actually a js framework)
   that is opened in automatically in browser (for command line)

##With jypyter notebook

#To save jupyter nootbook as pdf, install pandoc and MikTex 
# https://github.com/jgm/pandoc/releases/latest
# https://miktex.org/download
# then start jupyter - 
#Note first time while saving, it may download many files, hence may fail couple of time, but keep on trying 

$ jupyter notebook  #start server 

#New Notbook in jupyter :
    #Select a directory  and then New -> NoteBooks-Python2/3 in browser 

#in cell , execute by arrow mark 
quickref 
help 

#installation 
!pip install packagename

#reload all modules 
%load_ext autoreload
%autoreload 2


###Plotly - Quick Intro 

#only for online 
import plotly.plotly as py #for online 

#only for offline 
import plotly.offline as py 
py.init_notebook_mode(connected=True) #only in jupyter 

#Common steps 
import plotly.graph_objs as go
   
trace = {'x':[1,2],'y':[1,2]}
data = [trace]
layout = {}
fig = go.Figure(data=data,layout=layout)

#For command line 
plot_url = py.plot(fig)

#for jupyter notebook
py.iplot(fig)

##Plotly - Quick Intro - Data structure  
#Figure is json object
#figure is a dict containing 'data' which is  list of dict(trace) having keyword x,y,z etc 
#Note all objects are dict 

#plotly can understand simple data structure formatted according to below 
figure = { [ {x:..,y:..}, {x:[...], y:[...]},...] }  #simple structure ,default type='scatter'
#complex structure with many attributes https://plot.ly/python/reference/
#Note xaxis, yaxis can be part of 'layout' instead
figure = { 'data': [ {type: 'scatter', 
                            x:[..] ,
                            y:[..],
                           'line': {'color':'rgb(r,g,b)', 'width': n, 'shape':'linear','dash':'solid', },
                           'marker':{'symbol':'circle', 'size':n , 'opacity':.75 },
                           'name': "Plot name",
                           'mode': "lines+markers+text", 
                            },
                     {type:.... },
                     ...
                   ],
           'layout':{'font':{'family': ,'size':n,'color':"#444"}, 'title':"Plot layout Title",
                        'width':w,'height':h,'plot_bgcolor':'#FFF','showlegend':True ,
                        'legend':{'x':1,'y':1, 'bgcolor':, 'bordercolor':, 'font'{'size':n, 'color': }, },
                        'annotations' :[ {'text': ".", 'showarrow':True,'x': ,'y': , 'xref':'paper', 'yref':'paper'},
                            {}, ..],
                        },
            'xaxis':{'visible':True, 'color':'#444', 'title':"..", 
                     'type': ,'range':[min,max], 'nticks':n, 
                     'tickcolor':, 'ticks':, 'showgrid':True, 
                     'gridcolor':,'#444','showline'=True, mirror='all'},
            'yaxis':{same as 'xaxis'},  }
#Details             
x , y 
    list, numpy array, or Pandas series of numbers, strings, or datetimes.
type
    "scatter","bar","box","heatmap","histogram","histogram2d","area","pie","contour","histogram2dcontour"
    "scatter3d","surface","mesh3d","scattergeo","choropleth","scattergl" 
scatter.mode
    "lines", "markers", "lines+markers", "lines+markers+text", "none"
    Determines the drawing mode for this scatter trace. 
    If the provided `mode` includes "text" then the `text` elements appear at the coordinates. 
    Otherwise, the `text` elements appear on hover. 
    If there are less than 20 points, then the default is "lines+markers". 
    Otherwise, "lines".    
scatter.line.shape ( enumerated : "linear" | "spline" | "hv" | "vh" | "hvh" | "vhv" )
    default: "linear"
    Determines the line shape. With "spline" the lines are drawn using spline interpolation. The other available values correspond to step-wise line shapes.
scatter.line.dash 
   default: "solid"
   Sets the dash style of lines. Set to a dash type string ("solid", "dot", "dash", "longdash", "dashdot", or "longdashdot") or a dash length list in px (eg "5px,10px,2px,2px").
scatter.marker.symbol ( enumerated or array of enumerateds : "0" | "circle" | "100" | "circle-open" | "200" | "circle-dot" | "300" | "circle-open-dot" | "1" | "square" | "101" | "square-open" | "201" | "square-dot" | "301" | "square-open-dot" | "2" | "diamond" | "102" | "diamond-open" | "202" | "diamond-dot" | "302" | "diamond-open-dot" | "3" | "cross" | "103" | "cross-open" | "203" | "cross-dot" | "303" | "cross-open-dot" | "4" | "x" | "104" | "x-open" | "204" | "x-dot" | "304" | "x-open-dot" | "5" | "triangle-up" | "105" | "triangle-up-open" | "205" | "triangle-up-dot" | "305" | "triangle-up-open-dot" | "6" | "triangle-down" | "106" | "triangle-down-open" | "206" | "triangle-down-dot" | "306" | "triangle-down-open-dot" | "7" | "triangle-left" | "107" | "triangle-left-open" | "207" | "triangle-left-dot" | "307" | "triangle-left-open-dot" | "8" | "triangle-right" | "108" | "triangle-right-open" | "208" | "triangle-right-dot" | "308" | "triangle-right-open-dot" | "9" | "triangle-ne" | "109" | "triangle-ne-open" | "209" | "triangle-ne-dot" | "309" | "triangle-ne-open-dot" | "10" | "triangle-se" | "110" | "triangle-se-open" | "210" | "triangle-se-dot" | "310" | "triangle-se-open-dot" | "11" | "triangle-sw" | "111" | "triangle-sw-open" | "211" | "triangle-sw-dot" | "311" | "triangle-sw-open-dot" | "12" | "triangle-nw" | "112" | "triangle-nw-open" | "212" | "triangle-nw-dot" | "312" | "triangle-nw-open-dot" | "13" | "pentagon" | "113" | "pentagon-open" | "213" | "pentagon-dot" | "313" | "pentagon-open-dot" | "14" | "hexagon" | "114" | "hexagon-open" | "214" | "hexagon-dot" | "314" | "hexagon-open-dot" | "15" | "hexagon2" | "115" | "hexagon2-open" | "215" | "hexagon2-dot" | "315" | "hexagon2-open-dot" | "16" | "octagon" | "116" | "octagon-open" | "216" | "octagon-dot" | "316" | "octagon-open-dot" | "17" | "star" | "117" | "star-open" | "217" | "star-dot" | "317" | "star-open-dot" | "18" | "hexagram" | "118" | "hexagram-open" | "218" | "hexagram-dot" | "318" | "hexagram-open-dot" | "19" | "star-triangle-up" | "119" | "star-triangle-up-open" | "219" | "star-triangle-up-dot" | "319" | "star-triangle-up-open-dot" | "20" | "star-triangle-down" | "120" | "star-triangle-down-open" | "220" | "star-triangle-down-dot" | "320" | "star-triangle-down-open-dot" | "21" | "star-square" | "121" | "star-square-open" | "221" | "star-square-dot" | "321" | "star-square-open-dot" | "22" | "star-diamond" | "122" | "star-diamond-open" | "222" | "star-diamond-dot" | "322" | "star-diamond-open-dot" | "23" | "diamond-tall" | "123" | "diamond-tall-open" | "223" | "diamond-tall-dot" | "323" | "diamond-tall-open-dot" | "24" | "diamond-wide" | "124" | "diamond-wide-open" | "224" | "diamond-wide-dot" | "324" | "diamond-wide-open-dot" | "25" | "hourglass" | "125" | "hourglass-open" | "26" | "bowtie" | "126" | "bowtie-open" | "27" | "circle-cross" | "127" | "circle-cross-open" | "28" | "circle-x" | "128" | "circle-x-open" | "29" | "square-cross" | "129" | "square-cross-open" | "30" | "square-x" | "130" | "square-x-open" | "31" | "diamond-cross" | "131" | "diamond-cross-open" | "32" | "diamond-x" | "132" | "diamond-x-open" | "33" | "cross-thin" | "133" | "cross-thin-open" | "34" | "x-thin" | "134" | "x-thin-open" | "35" | "asterisk" | "135" | "asterisk-open" | "36" | "hash" | "136" | "hash-open" | "236" | "hash-dot" | "336" | "hash-open-dot" | "37" | "y-up" | "137" | "y-up-open" | "38" | "y-down" | "138" | "y-down-open" | "39" | "y-left" | "139" | "y-left-open" | "40" | "y-right" | "140" | "y-right-open" | "41" | "line-ew" | "141" | "line-ew-open" | "42" | "line-ns" | "142" | "line-ns-open" | "43" | "line-ne" | "143" | "line-ne-open" | "44" | "line-nw" | "144" | "line-nw-open" )
    default: "circle"
    Sets the marker symbol type. Adding 100 is equivalent to appending "-open" to a symbol name. Adding 200 is equivalent to appending "-dot" to a symbol name. Adding 300 is equivalent to appending "-open-dot" or "dot-open" to a symbol name.
layout.legend.x,y (number between or equal to -2 and 3)
    default: 1.02
    Sets the x position (in normalized coordinates) of the legend.
axis.type ( enumerated : "-" | "linear" | "log" | "date" | "category" )
    default: "-"
    Sets the axis type. By default, plotly attempts to determined the axis type by looking into the data of the traces that referenced the axis in question.
axis.nticks (integer greater than or equal to 0)
    default: 0
    Specifies the maximum number of ticks for the particular axis. The actual number of ticks will be chosen automatically to be less than or equal to `nticks`. Has an effect only if `tickmode` is set to "auto".
axis.ticks ( enumerated : "outside" | "inside" | "" )
    Determines whether ticks are drawn or not. 
    If "", this axis' ticks are not drawn. If "outside" ("inside"), this axis' are drawn outside (inside) the axis lines.
axis.tickcolor (color)
    default: "#444"
    Sets the tick color.
axis.mirror ( enumerated : True | "ticks" | False | "all" | "allticks" )
    Determines if the axis lines or/and ticks are mirrored to the opposite side of the plotting area. 
    If "True", the axis lines are mirrored. 
    If "ticks", the axis lines and ticks are mirrored. 
    If "False", mirroring is disable. 
    If "all", axis lines are mirrored on all shared-axes subplots. 
    If "allticks", axis lines and ticks are mirrored on all shared-axes subplots.
annotation.xref ( enumerated : "paper" | "/^x([2-9]|[1-9][0-9]+)?$/" )
    Sets the annotation's x coordinate axis. 
    If set to an x axis id (e.g. "x" or "x2"), 
    the `x` position refers to an x coordinate 
    If set to "paper", the `x` position refers to the distance 
    from the left side of the plotting area in normalized coordinates 
    where 0 (1) corresponds to the left (right) side.
annotation.yref ( enumerated : "paper" | "/^y([2-9]|[1-9][0-9]+)?$/" )
    Sets the annotation's y coordinate axis. 
    If set to an y axis id (e.g. "y" or "y2"), 
    the `y` position refers to an y coordinate 
    If set to "paper", the `y` position refers to the distance 
    from the bottom of the plotting area in normalized coordinates 
    where 0 (1) corresponds to the bottom (top).
annotation.x,y (number or categorical coordinate string)
    Sets the annotation's x position. 
    If the axis `type` is "log", then you must take the log of your desired range. 
    If the axis `type` is "date", it should be date strings, like date data, 
    though Date objects and unix milliseconds will be accepted 
    and converted to strings. 
    If the axis `type` is "category", it should be numbers, 
    using the scale where each category is assigned a serial number 
    from zero in the order it appears.


##Plotly - Quick Intro  - data structure in hierarchy
figure {}
    data []
        TRACE {}
            type ABC 
            x,y,z []
            color,text,size []
            colorscale ABC or []
            marker {}
                color ABC or []
                symbol ABC 
                size 123 or []
                line {}   #for marker line 
                    color ABC 
                    width 123 
                    
    layout {}
        title ABC 
        xaxis, yaxis {}
        scene {}
            xaxis, yaxis, zaxis {}
        geo {}
        legend {}
        annotations [{},..]
        shapes {}
    xaxis, yaxis {}
    
#meaning 
lowercase = key 
{} = dictionary 
[] = list 
ABC = a string 
123 = a number 

##Plotly - Quick Intro - plotly.graph_objs 
#has many classes encapsulating these structure for better abstraction
#Example 
data = [
    go.Scatter(                         # all "scatter" attributes: https://plot.ly/python/reference/#scatter
        x=[1, 2, 3],                    # more about "x":  /python/reference/#scatter-x
        y=[3, 1, 6],                    # more about "y":  /python/reference/#scatter-y
        marker=dict(                    # marker is an dict, marker keys: /python/reference/#scatter-marker
            color="rgb(16, 32, 77)"     # more about marker's "color": /python/reference/#scatter-marker-color
        )
    ),
    go.Bar(                         # all "bar" chart attributes: /python/reference/#bar
        x=[1, 2, 3],                # more about "x": /python/reference/#bar-x
        y=[3, 1, 6],                # /python/reference/#bar-y
        name="bar chart example"    # /python/reference/#bar-name
    )
]

layout = go.Layout(             # all "layout" attributes: /python/reference/#layout
    title="simple example",     # more about "layout's" "title": /python/reference/#layout-title
    xaxis=dict(                 # all "layout's" "xaxis" attributes: /python/reference/#layout-xaxis
        title="time"            # more about "layout's" "xaxis's" "title": /python/reference/#layout-xaxis-title
    ),
    annotations=[
        dict(                            # all "annotation" attributes: /python/reference/#layout-annotations
            text="simple annotation",    # /python/reference/#layout-annotations-text
            x=0,                         # /python/reference/#layout-annotations-x
            xref="paper",                # /python/reference/#layout-annotations-xref
            y=0,                         # /python/reference/#layout-annotations-y
            yref="paper"                 # /python/reference/#layout-annotations-yref
        )
    ]
)

figure = go.Figure(data=data, layout=layout)
py.plot(figure, filename='api-docs/reference-graph')

#Equivalent pure json objects 
data = [
    {
        'type': 'scatter',  # all "scatter" attributes: https://plot.ly/javascript/reference/#scatter
        'x': [1, 2, 3],     # more about "x": #scatter-x
        'y': [3, 1, 6],     # #scatter-y
        'marker': {         # marker is an object, valid marker keys: #scatter-marker
            'color': 'rgb(16, 32, 77)' # more about "marker.color": #scatter-marker-color
        }
    },
    {
        'type': 'bar',      # all "bar" chart attributes: #bar
        'x': [1, 2, 3],     # more about "x": #bar-x
        'y': [3, 1, 6],     # #bar-y
        'name': 'bar chart example' # #bar-name
    }
];

layout = {                      # all "layout" attributes: #layout
    'title': 'simple example',  # more about "layout.title": #layout-title
    'xaxis': {                  # all "layout.xaxis" attributes: #layout-xaxis
        'title': 'time'         # more about "layout.xaxis.title": #layout-xaxis-title
    },
    'annotations': [            # all "annotation" attributes: #layout-annotations
        {
            'text': 'simple annotation',    # #layout-annotations-text
            'x': 0,                         # #layout-annotations-x
            'xref': 'paper',                # #layout-annotations-xref
            'y': 0,                         # #layout-annotations-y
            'yref': 'paper'                 # #layout-annotations-yref
        }
    ]
}
figure = dict(data=data, layout=layout)


##Plotly - Quick Intro - plotly.graph_objs  - Other types 
type ("scatter")                    plotly.graph_objs.Scatter
type ("bar")                        plotly.graph_objs.Bar
type ("box")                        plotly.graph_objs.Box
type ("heatmap")                    plotly.graph_objs.Heatmap
type ("histogram")                  plotly.graph_objs.Histogram
type ("histogram2d")                plotly.graph_objs.Histogram2D
type ("area")                       plotly.graph_objs.Area
type ("pie")                        plotly.graph_objs.Pie
type ("contour")                    plotly.graph_objs.Contour
type ("histogram2dcontour")         plotly.graph_objs.Histogram2Dcontour
type ("scatter3d")                  plotly.graph_objs.Scatter3D
type ("surface")                    plotly.graph_objs.Surface
type ("mesh3d")                     plotly.graph_objs.Mesh3D
type ("scattergeo")                 plotly.graph_objs.Scattergeo
type ("choropleth")                 plotly.graph_objs.Choropleth
type ("scattergl")                  plotly.graph_objs.Scattergl

##Plotly - Quick Intro - plotly.tools.FigureFactory 
#FigureFactory has many types 
#http://takwatanabe.me/data_science/plotly_FF/top-plotly-FF.html
#Important Ones 
        create_2D_density
        create_dendrogram
        create_distplot
        create_scatterplotmatrix
        create_table
        create_violin
#All figure factories 
        create_2D_density
        create_annotated_heatmap
        create_candlestick
        create_dendrogram
        create_distplot
        create_gantt
        create_ohlc
        create_quiver
        create_scatterplotmatrix
        create_streamline
        create_table
        create_trisurf
        create_violin



###Plotly- plotly.offline.iplot/plot  - offline Mode 
#important attributes , filename and image for saving plot to other format 

plotly.offline.plot(figure_or_data, show_link=True, link_text='Export to plot.ly', validate=True, 
            output_type='file', include_plotlyjs=True, filename='temp-plot.html', auto_open=True, 
            image=None, image_filename='plot_image', image_width=800, image_height=600, config=None)
    Create a plotly graph locally as an HTML document or string.
    
plotly.offline.iplot(figure_or_data, show_link=True, link_text='Export to plot.ly', 
            validate=True, image=None, filename='plot_image', 
            image_width=800, image_height=600, config=None)
    Draw plotly graphs inside an IPython or Jupyter notebook without
    connecting to an external server.
    
    ** To save the chart to Plotly Cloud or Plotly Enterprise, use plotly.plotly.iplot.
   
    ** To embed an image of the chart, use plotly.image.ishow.
    
    figure_or_data -- a plotly.graph_objs.Figure or plotly.graph_objs.Data or
                      dict or list that describes a Plotly graph.
    
    Keyword arguments:
    show_link (default=True) -- display a link in the bottom-right corner of
                                of the chart that will export the chart to
                                Plotly Cloud or Plotly Enterprise
    link_text (default='Export to plot.ly') -- the text of export link
    validate (default=True) -- validate that all of the keys in the figure
                               are valid? omit if your version of plotly.js
                               has become outdated with your version of
                               graph_reference.json or if you need to include
                               extra, unnecessary keys in your figure.
    image (default=None |'png' |'jpeg' |'svg' |'webp') -- This parameter sets
        the format of the image to be downloaded, if we choose to download an
        image. This parameter has a default value of None indicating that no
        image should be downloaded. Please note: for higher resolution images
        and more export options, consider making requests to our image servers.
        Type: help(py.image) for more details.
    filename (default='plot') -- Sets the name of the file your image
        will be saved to. The extension should not be included.
    image_height (default=600) -- Specifies the height of the image in px.
    image_width (default=800) -- Specifies the width of the image in px.
    config (default=None) -- Plot view options dictionary. Keyword arguments
        show_link and link_text set the associated options in this
        dictionary if it doesn't contain them already.
    
    #Example:
    
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode()
    iplot([{'x': [1, 2, 3], 'y': [5, 2, 7]}])
    # We can also download an image of the plot by setting the image to the
    format you want. e.g. image='png'
    iplot([{'x': [1, 2, 3], 'y': [5, 2, 7]}], image='png')

    
###Plotly- help(plotly.iplot/plot)  - online mode 

plotly.plot(figure_or_data, validate=True, **plot_options)
plotly.iplot(figure_or_data, **plot_options)
    Create a unique url for this plot in Plotly and open in IPython.
    
    figure_or_data -- a plotly.graph_objs.Figure or plotly.graph_objs.Data or
                      dict or list that describes a Plotly graph.
                      See https://plot.ly/python/ for examples of
                      graph descriptions.
    
    plot_options keyword agruments:
    filename (string) -- the name that will be associated with this figure
    fileopt ('new' | 'overwrite' | 'extend' | 'append')
        - 'new': create a new, unique url for this plot
        - 'overwrite': overwrite the file associated with filename with this
        - 'extend': add additional numbers (data) to existing traces
        - 'append': add additional traces to existing data lists
    sharing ('public' | 'private' | 'secret') -- Toggle who can view this graph
        - 'public': Anyone can view this graph. It will appear in your profile
                    and can appear in search engines. You do not need to be
                    logged in to Plotly to view this chart.
        - 'private': Only you can view this plot. It will not appear in the
                     Plotly feed, your profile, or search engines. You must be
                     logged in to Plotly to view this graph. You can privately
                     share this graph with other Plotly users in your online
                     Plotly account and they will need to be logged in to
                     view this plot.
        - 'secret': Anyone with this secret link can view this chart. It will
                    not appear in the Plotly feed, your profile, or search
                    engines. If it is embedded inside a webpage or an IPython
                    notebook, anybody who is viewing that page will be able to
                    view the graph. You do not need to be logged in to view
                    this plot.
    world_readable (default=True) -- Deprecated: use "sharing".
                                     Make this figure private/public


###Plotly - Quick Into - Legends and axis 
trace1 = go.Scatter(name='Calvin', x=[1,2], y=[1,2])
trace2 = go.Scatter(name='Hobbs', x=[1,2], y=[2,1])

layout =  go.Layout(showlegend=True, legend=dict(x=0.2, y=0.5))
template = dict(showgrid=False, zeroline=False, nticks=20, showline=True, title='X Axis', mirror='all')

fig = go.Figure(data=[trace1, trace2], layout=layout, xaxis= template, yaxis=template)
py.iplot(fig)



##Line Plots : defalt scatter plot 
trace1 = go.Scatter(x=[1,2], y=[1,2])
trace2 = go.Scatter(x=[1,2], y=[2,1])
py.iplot([trace1, trace2])

##Scatter Plot 
trace1 = go.Scatter( x=[1,2,3], y=[1,2,3], text=['A', 'B', 'C'], textposition='top center', mode='markers+text')
py.iplot([trace1])

##Bar Plot 
trace1 = go.Bar( x=[1,2,3], y=[1,2,3])
py.iplot([trace1])

##Bubble Chart 
trace1 = go.Scatter( x=[1,2,3], y=[1,2,3], marker=dict(color=['red', 'blue', 'green'], size=[30,80,200]), mode='markers')
py.iplot([trace1])

##HeatMaps 
trace1 = go.Heatmap(z=[[1,2,3,4],[5,6,7,8]])
py.iplot([trace1])

##AreaMap 
trace1 = go.Scatter( x=[1,2,3], y=[1,2,3], fill='tonexty')
py.iplot([trace1])


##Box plot 
trace1 = go.Box(x=[1,2,3,4,5])
py.iplot([trace1])

#Histogram 
trace1 = go.Histogram(x=[1,2,3,4,5])
py.iplot([trace1])


##2D histogram 
trace1 = go.Histogram2d(x=[1,2,3,4,5], y=[1,2,3,3,4,5,6])
py.iplot([trace1])

##3D Line plots 
trace1 = go.Scatter3D(x=[1,2,3,4,5], y=[1,2,3,4,5], z=[1,2,3,4,5]. mode='lines')
py.iplot([trace1])

##3D scatter plots 
trace1 = go.Scatter3D(x=[1,2,3,4,5], y=[1,2,3,4,5], z=[1,2,3,4,5]. mode='markers')
py.iplot([trace1])


##Bubble Map 
trace = dict( type = 'scattergeo' lon=[100,400], lat=[0,0], marker=dict(marker=['red', 'blue'], size=[30,50]),mode='markers' )
py.iplot([trace])

##Choropleth Map 
trace = dict( type = 'choropleth' , locations=['AZ','CA','VE'], locationmode='USA-states', colorscale=['Viridis'], z=[10,20,40])
lyt=dict(geo=dict(scope='usa'))
map=go.Figure(data=[trace],layout=lyt)
py.iplot(map)


##Scatter Map 
trace = dict( type = 'scattergeo' lon=[100,400], lat=[0,0], marker=['Rome','Greece'],mode='markers' )
py.iplot([trace])





###Plotly- Save image in offline 

#Instead of saving the graphs to a server, your data and graphs will remain in your local system. 
#When your ready to share, you can just publish them to the web with an online Plotly account or to your company's internal Plotly Enterprise. 
#Else, you get error 
Aw, snap! We don't have an account for ''. Want to try again? You can authenticate with your email address or username. Sign in is not case sensitive.

Don't have an account? plot.ly

Questions? support@plot.ly

#offline mode 
from plotly import __version__
print __version__ # requires version >= 1.9.0



#plotly.offline.plot parameter:
#image (default=None |'png' |'jpeg' |'svg' |'webp')
py.plot(fig,image = 'png', image_filename='plot_image' , output_type='file', image_width=800, image_height=600, filename='temp-plot.html')
#since the output image is tied to HTML, it will open in browser and then downloads the image 


#example - in jupyter
import plotly.offline as offline
import plotly.graph_objs as go

offline.init_notebook_mode()  #for jupyter 

offline.iplot({'data': [{'y': [4, 2, 3, 4]}],
               'layout': {'title': 'Test Plot',
                          'font': dict(size=16)}},
             image='png', image_filename='newfile')


#in command line 

import plotly.offline as offline
import plotly.graph_objs as go

offline.plot({'data': [{'y': [4, 2, 3, 4]}],
               'layout': {'title': 'Test Plot',
                          'font': dict(size=16)}},
             image='png', image_filename='newfile')
             


             
###Plotly- To save the image(online)
#you need to login to plotly using your credentials (username and password)
#and get api_key
#not for offline , only online and saving is possible 
import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in('ndas1971', api_key='NES55JpLGgnIv5m3gXU6') # Replace the username, and API key with your credentials.

trace = go.Bar(x=[2, 4, 6], y= [10, 12, 15])
data = [trace]
layout = go.Layout(title='A Simple Plot', width=800, height=640)
fig = go.Figure(data=data, layout=layout)

py.image.save_as(fig, filename='a-simple-plot.png') #Add the appropriate extension  py.image.save_as(fig, 'chris-plot.jpg')

from IPython.display import Image
Image('a-simple-plot.png')

py.iplot(fig, filename='first_png')  #save it to online with filename='first_png' , get 'file_id' from share link in plotly site 


#Embed Static Images in Jupyter Notebooks(note Out would not be printed in pdf file, hence any fig must be embedded)
#not for offline 
py.image.ishow(fig)


###Plotly- Retrieve an Image from an Existing Online Chart (owner and file_id)
fig = py.get_figure('ndas1971', '1')  #OR fig = get_figure('https://plot.ly/~ndas1971/1')
fig['layout']['title'] = "New Title"
py.image.save_as(fig,'ndas-plot.png')
from IPython.display import Image
Image('ndas-plot.png') # Display a static image

###Plotly - color specification - from HTML specification 
1.hex (e.g. '#d3d3d3')
2.rgb (e.g. 'rgb(255, 0, 0)')
3.rgba (e.g. 'rgb(255, 0, 0, 0.5)')
4.hsl (e.g. 'hsl(0, 100%, 50%)')
5.hsv (e.g. 'hsv(0, 100%, 100%)')
6.named colors ('red', 'black'etc, check from https://www.w3.org/TR/css-color-3/)

#Get various colors 
$ pip install colorlover


import colorlover as cl
>>> cl.scales.keys()
dict_keys(['8', '11', '7', '3', '9', '6', '5', '4', '12', '10'])
>>> cl.scales['3'].keys()
dict_keys(['seq', 'div', 'qual'])
>>> cl.scales['3']['div'].keys()
dict_keys(['PuOr', 'RdGy', 'RdYlGn', 'PRGn', 'RdBu', 'RdYlBu', 'BrBG', 'Spectral', 'PiYG'])
>>> cl.scales['3']['div']['Spectral']
['rgb(252,141,89)', 'rgb(255,255,191)', 'rgb(153,213,148)']

cl.to_rgb( scale )
    Convert an HSL or numeric RGB color scale to string RGB color scale
>>> cl.to_rgb( cl.scales['3']['div']['RdYlBu'] )
['rgb(252,141,89)', 'rgb(255,255,191)', 'rgb(145,191,219)']

cl.to_html( scale )
    Traverse color scale dictionary and return available color scales as HTML string
>>> cl.to_html( cl.scales['3']['div']['RdYlBu'] )
'<div style="background-color:rgb(252,141,89);height:20px;width:20px;display:inline-block;"></div><div style="background-color:rgb(255,255,191);height:20px;width:20px;display:inline-block;"></div><div style="background-color:rgb(145,191,219);height:20px;width:20px;display:inline-block;"></div>'

cl.flipper( scale=None )
    Return the inverse of the color scale dictionary cl.scale
>>> cl.flipper()['div']['3']['RdYlBu']
['rgb(252,141,89)', 'rgb(255,255,191)', 'rgb(145,191,219)']



#All colors in cl.scales
# (in IPython notebook)
from IPython.display import HTML
HTML(cl.to_html( cl.scales ))

#Color interpolation 
bupu = cl.scales['9']['seq']['BuPu']
HTML( cl.to_html(bupu) )

bupu500 = cl.interp( bupu, 500 ) # Map color scale to 500 bins
HTML( cl.to_html( bupu500 ) )

data = Data([ Scatter(
    x = [ i * 0.1 for i in range(500) ],
    y = [ math.sin(j * 0.1) for j in range(500) ],
    mode='markers',
    marker=Marker(color=bupu500,size=22.0,line=Line(color='black',width=2)),
    text = cl.to_rgb( bupu500 ),
    opacity = 0.7
)])
layout = Layout( showlegend=False, xaxis=XAxis(zeroline=False), yaxis=YAxis(zeroline=False) )
fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='spectral_bubblechart')








###Plotly- Subplots and usage of fig.append_trace() 

plotly.tools.get_subplots(rows=1, columns=1, print_grid=False, **kwargs):
    Return a dictionary instance with the subplots set in 'layout'.
    rows (kwarg, int greater than 0, default=1):
        Number of rows, evenly spaced vertically on the figure.
    columns (kwarg, int greater than 0, default=1):
        Number of columns, evenly spaced horizontally on the figure.
    horizontal_spacing (kwarg, float in [0,1], default=0.1):
        Space between subplot columns. Applied to all columns.
    vertical_spacing (kwarg, float in [0,1], default=0.05):
        Space between subplot rows. Applied to all rows.
    print_grid (kwarg, True | False, default=False):
        If True, prints a tab-delimited string representation
        of your plot grid.
    horizontal_spacing (kwarg, float in [0,1], default=0.2 / columns):
        Space between subplot columns.
    vertical_spacing (kwarg, float in [0,1], default=0.3 / rows):
        Space between subplot rows.
        
#Example 1:
# stack two subplots vertically
fig = tools.get_subplots(rows=2)
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x1', yaxis='y1')]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

#Example 2:
# print out string showing the subplot grid you've put in the layout
fig = tools.get_subplots(rows=3, columns=2, print_grid=True)

    


plotly.tools.make_subplots(rows=1, cols=1,
                  shared_xaxes=False, shared_yaxes=False,
                  start_cell='top-left', print_grid=True,
                  **kwargs):
    Return an instance of plotly.graph_objs.Figure  with the subplots domain set in 'layout'.
    rows (kwarg, int greater than 0, default=1):
        Number of rows in the subplot grid.
    cols (kwarg, int greater than 0, default=1):
        Number of columns in the subplot grid.
    shared_xaxes (kwarg, boolean or list, default=False)
        Assign shared x axes.
        If True, subplots in the same grid column have one common
        shared x-axis at the bottom of the gird.
        To assign shared x axes per subplot grid cell (see 'specs'),
        send list (or list of lists, one list per shared x axis)
        of cell index tuples.
    shared_yaxes (kwarg, boolean or list, default=False)
        Assign shared y axes.
        If True, subplots in the same grid row have one common
        shared y-axis on the left-hand side of the gird.
        To assign shared y axes per subplot grid cell (see 'specs'),
        send list (or list of lists, one list per shared y axis)
        of cell index tuples.

    start_cell (kwarg, 'bottom-left' or 'top-left', default='top-left')
        Choose the starting cell in the subplot grid used to set the
        domains of the subplots.
    print_grid (kwarg, boolean, default=True):
        If True, prints a tab-delimited string representation of
        your plot grid.
    horizontal_spacing (kwarg, float in [0,1], default=0.2 / cols):
        Space between subplot columns.
        Applies to all columns (use 'specs' subplot-dependents spacing)
    vertical_spacing (kwarg, float in [0,1], default=0.3 / rows):
        Space between subplot rows.
        Applies to all rows (use 'specs' subplot-dependents spacing)
    subplot_titles (kwarg, list of strings, default=empty list):
        Title of each subplot.
        "" can be included in the list if no subplot title is desired in
        that space so that the titles are properly indexed.
    specs (kwarg, list of lists of dictionaries):
        Subplot specifications.
        ex1: specs=[[{}, {}], [{'colspan': 2}, None]]
        ex2: specs=[[{'rowspan': 2}, {}], [None, {}]]
        - Indices of the outer list correspond to subplot grid rows
          starting from the bottom. The number of rows in 'specs'
          must be equal to 'rows'.
        - Indices of the inner lists correspond to subplot grid columns
          starting from the left. The number of columns in 'specs'
          must be equal to 'cols'.
        - Each item in the 'specs' list corresponds to one subplot
          in a subplot grid. (N.B. The subplot grid has exactly 'rows'
          times 'cols' cells.)
        - Use None for blank a subplot cell (or to move pass a col/row span)
        - Note that specs[0][0] has the specs of the 'start_cell' subplot.
        - Each item in 'specs' is a dictionary.
            The available keys are:
            * is_3d (boolean, default=False): flag for 3d scenes
            * colspan (int, default=1): number of subplot columns
                for this subplot to span.
            * rowspan (int, default=1): number of subplot rows
                for this subplot to span.
            * l (float, default=0.0): padding left of cell
            * r (float, default=0.0): padding right of cell
            * t (float, default=0.0): padding right of cell
            * b (float, default=0.0): padding bottom of cell
        - Use 'horizontal_spacing' and 'vertical_spacing' to adjust
          the spacing in between the subplots.
    insets (kwarg, list of dictionaries):
        Inset specifications.
        - Each item in 'insets' is a dictionary.
            The available keys are:
            * cell (tuple, default=(1,1)): (row, col) index of the
                subplot cell to overlay inset axes onto.
            * is_3d (boolean, default=False): flag for 3d scenes
            * l (float, default=0.0): padding left of inset
                  in fraction of cell width
            * w (float or 'to_end', default='to_end') inset width
                  in fraction of cell width ('to_end': to cell right edge)
            * b (float, default=0.0): padding bottom of inset
                  in fraction of cell height
            * h (float or 'to_end', default='to_end') inset height
                  in fraction of cell height ('to_end': to cell top edge)

#Example 1:
# stack two subplots vertically
fig = tools.make_subplots(rows=2)

#This is the format of  plot grid:
#[ (1,1) x1,y1 ]
#[ (2,1) x2,y2 ]

fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

#Example 2:
# subplots with shared x axes
fig = tools.make_subplots(rows=2, shared_xaxes=True)

#This is the format of your plot grid:
#[ (1,1) x1,y1 ]
#[ (2,1) x1,y2 ]


fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], yaxis='y2')]

#Example 3:
# irregular subplot layout (more examples below under 'specs')
#2x2 ie 4 subplots 
#specs=[[{}, {}], [{'colspan': 2}, None] means 
#specs of [0][0] is {}, of [0][1] is {} , {} means default specs 
#specs of [1][0] is {'colspan': 2} ie spanning two cols, of [1][1] is None, None means no column 

fig = tools.make_subplots(rows=2, cols=2,
                          specs=[ [{}, {}],
                                  [{'colspan': 2}, None]
                                  ])

#This is the format of your plot grid!
#[ (1,1) x1,y1 ]  [ (1,2) x2,y2 ]
#[ (2,1) x3,y3           -      ]

fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x3', yaxis='y3')]

#Example 4:
# insets
fig = tools.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])

#This is the format of your plot grid!
#[ (1,1) x1,y1 ]
#With insets:
#[ x2,y2 ] over [ (1,1) x1,y1 ]

fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

#Example 5:
# include subplot titles
fig = tools.make_subplots(rows=2, subplot_titles=('Plot 1','Plot 2'))

#This is the format of your plot grid:
#[ (1,1) x1,y1 ]
#[ (2,1) x2,y2 ]

fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

#Example 6:
# Include subplot title on one plot (but not all)
fig = tools.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}],
                          subplot_titles=('','Inset'))

#This is the format of your plot grid!
#[ (1,1) x1,y1 ]

#With insets:
#[ x2,y2 ] over [ (1,1) x1,y1 ]

fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2])]
fig['data'] += [Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2')]

    
#Example with fig.append_trace()
from plotly import tools
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

heatmap = go.Heatmap(
        z=[[1, 20, 30],
           [20, 1, 60],
           [30, 60, 1]],
        showscale=False
        )
        
#set up the trace object for our wind rose chart, 

y0 = np.random.randn(50)
y1 = np.random.randn(50)+1

box_1 = go.Box(
    y=y0
)
box_2 = go.Box(
    y=y1
)
data = [heatmap, box_1, box_2]

#2x2 ie 4 subplots 
#specs=[[{}, {}], [{'colspan': 2}, None] means 
#specs of [0][0] is {}, of [0][1] is {} , {} means default specs 
#specs of [1][0] is {'colspan': 2} ie spanning two cols, of [1][1] is None, None means no column 
#This is the format of your plot grid:
#[ (1,1) x1,y1 ]  [ (1,2) x2,y2 ]
#[ (2,1) x3,y3           -      ]

fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('First Subplot','Second Subplot', 'Third Subplot'))

fig.append_trace(box_1, 1, 1)  #this index starts from 1 
fig.append_trace(box_2, 1, 2)
fig.append_trace(heatmap, 2, 1)

fig['layout'].update(height=600, width=600, title='i <3 subplots')

py.iplot(fig, filename='box_heatmap1')


#fig object 
>>> fig
#output 
{'data': [{'type': 'box',
   'xaxis': 'x1',
   'y': array([-0.91373828,  1.20546486, -0.096304  ,  2.37294867,  0.75779673,
          -0.85565264, -0.21226529, -0.15192424,  0.6870414 , -0.90325114,
           0.77816331, -2.14903403,  0.23682143,  0.47138943, -0.26381095,
          -1.72174484,  0.70897631,  1.19831382,  0.6537401 , -0.17682102,
          -0.74564104, -0.97780188, -1.87756116, -0.29443922,  0.45824652,
          -0.80421393, -0.41012153, -1.04762769,  0.42932349,  1.2102797 ,
           0.10064935,  0.38721905, -1.69759714,  0.30360933,  0.18920693,
           0.88070821,  0.06445955,  0.2453985 ,  0.42510519, -0.03765171,
           0.47967058,  1.37685937, -0.72184219,  0.65808692, -0.36645823,
           0.52223803, -0.03703202,  0.09859009,  0.46216406,  0.90797845]),
   'yaxis': 'y1'},
  {'type': 'box',
   'xaxis': 'x2',
   'y': array([ 1.39114565,  2.42361719,  0.74492841,  0.5796092 ,  0.70920041,
           1.84970903,  1.28910587,  2.46293187,  0.46195123, -0.01825562,
           2.60459886,  0.03786005,  1.11958395, -0.20198951, -1.09348176,
           1.8939656 , -0.72453807,  1.45243595,  2.39173323,  1.56600484,
           1.78916462,  1.61941803,  2.25792077,  0.89589498,  1.0673083 ,
           1.0942342 ,  0.42048992, -0.35346571,  0.59505795,  2.17376683,
           2.13342539,  0.59371436,  0.32297912,  0.35487172, -0.19119881,
           1.43115097,  0.2614727 ,  0.90561162,  1.84856626,  1.49876699,
           0.3717146 ,  0.03332516,  0.16060302, -1.63113126,  0.28004198,
           0.59134313,  1.76526806, -0.04537056,  1.8133062 ,  0.98513375]),
   'yaxis': 'y2'},
  {'showscale': False,
   'type': 'heatmap',
   'xaxis': 'x3',
   'yaxis': 'y3',
   'z': [[1, 20, 30], [20, 1, 60], [30, 60, 1]]}],
 'layout': {'annotations': [{'font': {'size': 16},
    'showarrow': False,
    'text': 'First Subplot',
    'x': 0.225,
    'xanchor': 'center',
    'xref': 'paper',
    'y': 1.0,
    'yanchor': 'bottom',
    'yref': 'paper'},
   {'font': {'size': 16},
    'showarrow': False,
    'text': 'Second Subplot',
    'x': 0.775,
    'xanchor': 'center',
    'xref': 'paper',
    'y': 1.0,
    'yanchor': 'bottom',
    'yref': 'paper'},
   {'font': {'size': 16},
    'showarrow': False,
    'text': 'Third Subplot',
    'x': 0.5,
    'xanchor': 'center',
    'xref': 'paper',
    'y': 0.375,
    'yanchor': 'bottom',
    'yref': 'paper'}],
  'height': 600,
  'title': 'i <3 subplots',
  'width': 600,
  'xaxis1': {'anchor': 'y1', 'domain': [0.0, 0.45]},
  'xaxis2': {'anchor': 'y2', 'domain': [0.55, 1.0]},
  'xaxis3': {'anchor': 'y3', 'domain': [0.0, 1.0]},
  'yaxis1': {'anchor': 'x1', 'domain': [0.625, 1.0]},
  'yaxis2': {'anchor': 'x2', 'domain': [0.625, 1.0]},
  'yaxis3': {'anchor': 'x3', 'domain': [0.0, 0.375]}}}

#We actually see that the second boxplot has its own xaxis and yaxis. 
#Note : meaning of data.xaxis and data.yaxis 
# If 'x' (the default value), the x coordinates refer to layout.xaxis. 
#If 'x1', the x coordinates refer to layout.xaxis1 and so on 
#similarly for yaxis 

#To have same axis for box plot with the first subplot. 
#Then from the layout section, we should remove the additional xaxis and yaxis that are drawn for us. 

fig.data[1].yaxis = 'y1'
fig.data[1].xaxis = 'x1'
del fig.layout['xaxis2']
del fig.layout['yaxis2']


#Remove the annotation for box (Second Subplot title), 
#as well as the First Subplot title, 
#and then extend the range of xaxis1 to the entire plotting surface.

del fig.layout.annotations[0]   #deletes annotation for First Subplot
del fig.layout.annotations[0]   #deletes annotation for Second Subplot because of shift
fig.layout.xaxis1.domain = [0.0, 1]

py.iplot(fig, filename='box-heatmap-fixed')










@@@

                                     
                                    
###Plotly - Various Examples 

##Plotly -  Example - Displaying Table and multiple traces
import pandas as pd
import numpy as np
import scipy as sp
#import plotly.plotly as py #for online storage, but login is required at first 
import plotly.offline as py

py.init_notebook_mode(connected=True)#for IPython 


#display it in a table
import plotly.figure_factory as ff
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/school_earnings.csv")

table = ff.create_table(df)
py.iplot(table, filename='table1')  #filename associated with this plot (not physical file)

#Plotting Inline

from plotly.graph_objs import *

data = [Bar(x=df.School,
            y=df.gap)]

py.iplot(data, filename='basic_bar')  #for online mode, it would save the data in plotly


#Plotting multiple traces and styling the chart with custom colors and titles 

trace_women = Bar(x=df.School,
                  y=df.Women,
                  name='Women',
                  marker=dict(color='#ffcdd2'))

trace_men = Bar(x=df.School,
                y=df.Men,
                name='Men',
                marker=dict(color='#A2D5F2'))

trace_gap = Bar(x=df.School,
                y=df.gap,
                name='Gap',
                marker=dict(color='#59606D'))

data = [trace_women, trace_men, trace_gap]
layout = Layout(title="Average Earnings for Graduates",
                xaxis=dict(title='School'),
                yaxis=dict(title='Salary (in thousands)'))
fig = Figure(data=data, layout=layout)

py.iplot(fig, sharing='secret', filename='styled_bar')




##Plotly -  Example - 3D Plotting

import plotly.offline as py #import plotly.plotly as py

from plotly.graph_objs import *
#for for offline mode 
py.init_notebook_mode(connected=True)#for IPython 

import numpy as np

s = np.linspace(0, 2 * np.pi, 240)
t = np.linspace(0, np.pi, 240)
tGrid, sGrid = np.meshgrid(s, t)  #create 2D rectangular grid with dimension x=s and y=t 

r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
z = r * np.cos(tGrid)                  # z = r*cos(t)

surface = Surface(x=x, y=y, z=z)
data = Data([surface])

layout = Layout(
    title='Parametric Plot',
    scene=Scene(
        xaxis=XAxis(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=YAxis(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=ZAxis(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        )
    )
)

fig = Figure(data=data, layout=layout)
#py.iplot(fig, filename='parametric_plot') #for online saves to server
py.plot(fig, filename='parametric_plot') #opens browser 





  
##Plotly -  Example - Various Scatter Plot and WebGL

import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for offline
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# Create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')

# or plot with: 
#plot_url = py.plot(data, filename='basic-line')


#WebGL with Scattergl() in place of Scatter()

N = 100000
trace = go.Scattergl(
    x = np.random.randn(N),
    y = np.random.randn(N),
    mode = 'markers',
    marker = dict(
        color = 'FFBAD2',
        line = dict(width = 1)
    )
)
data = [trace]
py.iplot(data, filename='compare_webgl')


##Scatter with a Color Dimension

trace1 = go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size='16',
        color = np.random.randn(500), #set color equal to a variable
        colorscale='Viridis',
        showscale=True
    )
)
data = [trace1]

py.iplot(data, filename='scatter-plot-with-colorscale')



##Plotly -  Example - Line and Scatter Plots


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

N = 100
random_x = np.linspace(0, 1, N)
random_y0 = np.random.randn(N)+5
random_y1 = np.random.randn(N)
random_y2 = np.random.randn(N)-5

# Create traces
trace0 = go.Scatter(
    x = random_x,
    y = random_y0,
    mode = 'markers',#pure scatter plot
    name = 'markers'   
)
trace1 = go.Scatter(
    x = random_x,
    y = random_y1,
    mode = 'lines+markers',#line with marker 
    name = 'lines+markers'  
)
trace2 = go.Scatter(
    x = random_x,
    y = random_y2,
    mode = 'lines', #pure line plot
    name = 'lines'
)

data = [trace0, trace1, trace2]
py.iplot(data, filename='scatter-mode')








##Plotly -  Example - Simple Slider Control
#Sliders can now be used in Plotly to change the data displayed or style of a plot


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import numpy as np

data = [dict(
        visible = False,
        line=dict(color='00CED1', width=6),
        name = '𝜈 = '+str(step),
        x = np.arange(0,10,0.01),
        y = np.sin(step*np.arange(0,10,0.01))) 
            for step in np.arange(0,5,0.1)]  #create many plots, only make one Visible 
data[10]['visible'] = True

py.iplot(data, filename='Single Sine Wave')


steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',
        args = ['visible', [False] * len(data)],
    )
step['args'][1][i] = True # Toggle i'th trace to "visible"
steps.append(step)

sliders = [dict(
    active = 10,
    currentvalue = {"prefix": "Frequency: "},
    pad = {"t": 50},
    steps = steps
)]

layout = dict(sliders=sliders)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='Sine Wave Slider')




##Plotly -  Example - Multiple Subplots with Titles


from plotly import tools
import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

trace1 = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
trace2 = go.Scatter(x=[20, 30, 40], y=[50, 60, 70])
trace3 = go.Scatter(x=[300, 400, 500], y=[600, 700, 800])
trace4 = go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000])

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Plot 1', 'Plot 2',
                                                          'Plot 3', 'Plot 4'))
#output 
#This is the format of your plot grid:
#[ (1,1) x1,y1 ]  [ (1,2) x2,y2 ]  #note x* is translated to xaxis* under layout 
#[ (2,1) x3,y3 ]  [ (2,2) x4,y4 ]

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)

fig['layout'].update(height=600, width=600, title='Multiple Subplots' +
                                                  ' with Titles')

py.iplot(fig, filename='make-subplots-multiple-with-titles')




##Plotly -  Example - Multiple Subplots with various options 


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

trace1 = go.Scatter(
    x=[1, 2, 3],
    y=[4, 5, 6]
)
trace2 = go.Scatter(
    x=[20, 30, 40],
    y=[50, 60, 70],
    xaxis='x2',     #x*, y* means xaxis*/yaxis* under layout
    yaxis='y2'
)
trace3 = go.Scatter(
    x=[300, 400, 500],
    y=[600, 700, 800],
    xaxis='x3',
    yaxis='y3'
)
trace4 = go.Scatter(
    x=[4000, 5000, 6000],
    y=[7000, 8000, 9000],
    xaxis='x4',
    yaxis='y4'
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(      #xaxis, yaxis are named as 2,3,4 for other subplots 
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'    #y3 means yaxis3
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='multiple-subplots')




##Plotly -  Example - Stacked Subplots with a Shared X-Axis


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

trace1 = go.Scatter(
    x=[0, 1, 2],
    y=[10, 11, 12]
)
trace2 = go.Scatter(
    x=[2, 3, 4],
    y=[100, 110, 120],
    yaxis='y2'   #x*, y* means xaxis*/yaxis* under layout
)
trace3 = go.Scatter(
    x=[3, 4, 5],
    y=[1000, 1100, 1200],
    yaxis='y3'
)
data = [trace1, trace2, trace3]
layout = go.Layout(
    yaxis=dict(
        domain=[0, 0.33]
    ),
    legend=dict(
        traceorder='reversed'
    ),
    yaxis2=dict(
        domain=[0.33, 0.66]
    ),
    yaxis3=dict(
        domain=[0.66, 1]
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-subplots-shared-x-axis')


##Plotly -  Example - Simple Inset Graph


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

trace1 = go.Scatter(
    x=[1, 2, 3],
    y=[4, 3, 2]
)
trace2 = go.Scatter(
    x=[20, 30, 40],
    y=[30, 40, 50],
    xaxis='x2',  #x*, y* means xaxis*/yaxis* under layout
    yaxis='y2'
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2'
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-inset')



##Plotly -  Example - Mixed Subplot


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
from plotly.graph_objs import *

import pandas as pd

# read in volcano database data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/volcano_db.csv')

# frequency of Country
freq = df.copy()
freq = freq.Country.value_counts().reset_index().rename(columns={'index': 'x'})

# plot(1) top 10 countries by total volcanoes
locations = Bar(x=freq['x'][0:10],y=freq['Country'][0:10], marker=dict(color='#CF1020'))

# read in 3d volcano surface data
df_v = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv')

# plot(2) 3d surface of volcano
threed = Surface(z=df_v.values.tolist(), colorscale='Reds', showscale=False)

# plot(3)  scattergeo map of volcano locations
trace3 = {
  "geo": "geo3", 
  "lon": df['Longitude'],
  "lat": df['Latitude'],
  "hoverinfo": 'text',
  "marker": {
    "size": 4,
    "opacity": 0.8,
    "color": '#CF1020',
    "colorscale": 'Viridis'
  }, 
  "mode": "markers", 
  "type": "scattergeo"
}

data = Data([locations, threed, trace3])

# control the subplot below using domain in 'geo', 'scene', and 'axis'
layout = {
  "plot_bgcolor": 'black',
  "paper_bgcolor": 'black',
  "titlefont": {
      "size": 20,
      "family": "Raleway"
  },
  "font": {
      "color": 'white'
  },
  "dragmode": "zoom", 
  "geo3": {
    "domain": {
      "x": [0, 0.55], 
      "y": [0, 0.9]
    }, 
    "lakecolor": "rgba(127,205,255,1)",
    "oceancolor": "rgb(6,66,115)",
    "landcolor": 'white',
    "projection": {"type": "orthographic"}, 
    "scope": "world", 
    "showlakes": True,
    "showocean": True,
    "showland": True,
    "bgcolor": 'black'
  }, 
  "margin": {
    "r": 10, 
    "t": 25, 
    "b": 40, 
    "l": 60
  }, 
  "scene": {"domain": {
      "x": [0.5, 1], 
      "y": [0, 0.55]
    },
           "xaxis": {"gridcolor": 'white'},
           "yaxis": {"gridcolor": 'white'},
           "zaxis": {"gridcolor": 'white'}
           }, 
  "showlegend": False, 
  "title": "<br>Volcano Database", 
  "xaxis": {
    "anchor": "y", 
    "domain": [0.6, 0.95]
  }, 
  "yaxis": {
    "anchor": "x", 
    "domain": [0.65, 0.95],
    "showgrid": False
  }
}

annotations = { "text": "Source: NOAA",
               "showarrow": False,
               "xref": "paper",
               "yref": "paper",
               "x": 0,
               "y": 0}

layout['annotations'] = [annotations]

fig = Figure(data=data, layout=layout)
py.iplot(fig, filename = "Mixed Subplots Volcano")


##Plotly -  Example - Time Series Plot with datetime Objects


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

from datetime import datetime
import pandas_datareader.data as web

df = web.DataReader("aapl", 'yahoo',        #yahoo reader is broken , check pandas section
                    datetime(2015, 1, 1),
                    datetime(2016, 7, 1))

data = [go.Scatter(x=df.index, y=df.High)]

py.iplot(data)



##Plotly -  Example - Time Series Plot with Date Strings


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

data = [go.Scatter(
          x=df.Date,
          y=df['AAPL.Close'])]

py.iplot(data)



##Plotly -  Example - Time Series Plot with Rangeslider


import plotly.offline as py;py.init_notebook_mode(connected=True)#import plotly.plotly as py#for online mode
import plotly.graph_objs as go

import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

trace_high = go.Scatter(
    x=df.Date,
    y=df['AAPL.High'],
    name = "AAPL High",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

trace_low = go.Scatter(
    x=df.Date,
    y=df['AAPL.Low'],
    name = "AAPL Low",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8)

data = [trace_high,trace_low]

layout = dict(
    title='Time Series with Rangeslider',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Time Series with Rangeslider")


###Plotly -  Example - Plot with spark 
#https://plot.ly/python/apache-spark/
#Using IPython with pyspark 
#http://blog.cloudera.com/blog/2014/08/how-to-use-ipython-notebook-with-apache-spark/

$ ipython profile create pyspark

# profile directory ~/.ipython/profile_pyspark/. 
#Edit the file ~/.ipython/profile_pyspark/ipython_notebook_config.py 
c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8880 # or whatever you want; be aware of conflicts with SPARK

#Create the file ~/.ipython/profile_pyspark/startup/00-pyspark-setup.py 
import os
import sys

spark_home = os.environ.get('SPARK_HOME', None)
  if not spark_home:
      raise ValueError('SPARK_HOME environment variable is not set')
sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.1-src.zip'))
with open(os.path.join(spark_home, 'python/pyspark/shell.py')) as f:
    code = compile(f.read(), os.path.join(spark_home, 'python/pyspark/shell.py'), 'exec')
    exec(code)

##Starting IPython Notebook with PySpark
#IPython Notebook should be run on a machine from which PySpark would be run on, 

#Env variable - set in Win, export in Unix 
# for the CDH-installed Spark
set SPARK_HOME='/opt/cloudera/parcels/CDH/lib/spark'
# this is where you specify all the options you would normally add after bin/pyspark
set PYSPARK_SUBMIT_ARGS='--master yarn --deploy-mode client --num-executors 24 --executor-memory 10g --executor-cores 5'

#Then execute 
ipython notebook --profile=pyspark

#Check 
from __future__ import print_function #python 3 support
print(sc)

#All imports 
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import requests
requests.packages.urllib3.disable_warnings()
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import *
from pyspark.ml.linalg import *

#load data file 
btd = spark.read.format("json").load("data/btd2.json")
>>> btd.printSchema()
>>> btd.show()

spark.registerDataFrameAsTable(btd, "bay_area_bike")
df2 = spark.sql("SELECT Duration as d1 from bay_area_bike where Duration < 7200") #2 hrs

#Create plot 
data = go.Data([go.Histogram(x=df2.toPandas()['d1'])])  
py.iplot(data, filename="spark/less_2_hour_rides")

#Multiple plots 
df3 = spark.sql("SELECT Duration as d1 from bay_area_bike where Duration < 2000") #30 mins 

s1 = df2.sample(False, 0.05, 20)  #plot only 5%
s2 = df3.sample(False, 0.05, 2500)

data = go.Data([
        go.Histogram(x=s1.toPandas()['d1'], name="Large Sample"),
        go.Histogram(x=s2.toPandas()['d1'], name="Small Sample")
    ])

py.iplot(data, filename="spark/sample_rides")

#bike rentals from individual stations
dep_stations = btd.groupBy(btd['Start Station']).count().toPandas().sort('count', ascending=False)
dep_stations['Start Station'][:3] # top 3 stations
#plot 
def transform_df(df):
    df['counts'] = 1
    df['Start Date'] = df['Start Date'].apply(pd.to_datetime)
    return df.set_index('Start Date').resample('D', how='sum')

pop_stations = [] # being popular stations - we could easily extend this to more stations
for station in dep_stations['Start Station'][:3]:
    temp = transform_df(btd.where(btd['Start Station'] == station).select("Start Date").toPandas())
    pop_stations.append(
        go.Scatter(
        x=temp.index,
        y=temp.counts,
        name=station
        )
    )
    
data = go.Data(pop_stations)
py.iplot(data, filename="spark/over_time")