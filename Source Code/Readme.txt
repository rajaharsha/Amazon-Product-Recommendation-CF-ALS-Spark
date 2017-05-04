Step 1: Open Hortonworks Spark Virtual box/ spark environment with atleast Spark 1.2.1 and Python 2.7.7 installed.
Step 2: If required install numpy, scipy libraries.
Step 3: Execute the command "export SPARK_HOME=/usr/hdp/2.2.4.2-2/spark" in unix server.
Step 4: Transfer the code file AmazonALS.py, ratings.dat, products.dat, users.dat to unix server.
Step 5: Create a directory in HDFS like AZ/input and trandfer the .dat files to input HDFS directory created.
Step 6: To run the recommendation program execute: spark-submit AmazonALS.py <InputDirectory> <OutputFileName> <Iterations> <Partitions>
Step 7: If the number of iterations are more than 10, the program likely takes more than 15 minutes.
Step 8: An output files is created in the same directory with userID,recommendedProduct,predictedRating
