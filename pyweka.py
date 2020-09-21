import sys
import weka.core.jvm as jvm
import weka.core.packages as packages
from weka.core.classes import complete_classname
from weka.core.converters import Loader
from weka.classifiers import Classifier
import weka.plot.graph as graph  # NB: pygraphviz and PIL are required
from weka.core.classes import Random, from_commandline
import weka.core.serialization as serialization
import re
from tqdm import tqdm
import time
import pandas as pd

start_time = time.time()
jvm.start(packages=True, max_heap_size="12g") #max_heap_size 512m, 4g. packages=true searches for weka packages in installation program
#pkg = "PBC4cip"

# install package if necessary 
#if not packages.is_installed(pkg):1.0E 
#    print("Installing %s..." % pkg)
#    packages.install_package(pkg)
#    print("Installed %s, please re-run script!" % pkg)
#    jvm.stop()
#    sys.exit(0)

# testing classname completion
#print(complete_classname(".PBC4cip"))

print("\n\n\n\n\n")
print(">>> Start...")

data_dir = "D:/GoogleDrive/ITESM/3rd Semester/Tecnicas de ML/Assignment 3/"
arff_file = "universities.arff"

loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_dir + arff_file)
data.class_is_last()

filename_array = ["NumberOfTrees", "RandomFeatures", "DepthOfTrees", "MinimumObjectByLeaf", "RandomSubSpace", "InfoGainAttributeEval", "PrincipalComponents", "GainRatioAttributeEval"]
filename = str(filename_array[2])
filename_index = filename_array.index(filename)

cmdline = []

# Criterions

# Number of trees: 1000
trees = 1000
cmdline.append(f'weka.classifiers.trees.PBC4cip -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees {trees} -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth -1 \\\\\\\"-minimalObjByLeaf \\\\\\\" 2 -minimalSplitGain 1.0E-30\\\"\"')

# Random features
numberOfFeatures_array = [5, 10, 15]
numberOfFeatures = numberOfFeatures_array[0]
cmdline.append(f'weka.classifiers.trees.PBC4cip -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures {numberOfFeatures} -numTrees {trees} -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth -1 \\\\\\\"-minimalObjByLeaf \\\\\\\" 2 -minimalSplitGain 1.0E-30\\\"\"')

# Depth of trees
maxDepth_array = [2, 3, 4]
maxDepth = maxDepth_array[0]
cmdline.append(f'weka.classifiers.trees.PBC4cip -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees {trees} -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth {maxDepth} \\\\\\\"-minimalObjByLeaf \\\\\\\" 2 -minimalSplitGain 1.0E-30\\\"\"')

# Minimum object by leaf
objectsByLeaf_array = [3, 4, 5]
objectsByLeaf = objectsByLeaf_array[0]
cmdline.append(f'weka.classifiers.trees.PBC4cip -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees {trees} -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth -1 \\\\\\\"-minimalObjByLeaf \\\\\\\" {objectsByLeaf} -minimalSplitGain 1.0E-30\\\"\"')

# RandomSubSpace
randomSubSpace_array = [0.2 , 0.4, 0.6]
randomSubSpace = randomSubSpace_array[0]
cmdline.append(f'weka.classifiers.meta.RandomSubSpace -P {randomSubSpace} -S 1 -num-slots 1 -I 10 -W weka.classifiers.trees.PBC4cip -- -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees {trees} -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth -1 \\\\\\\"-minimalObjByLeaf \\\\\\\" 2 -minimalSplitGain 1.0E-30\\\"\"')

# Attribute selection algorithm - InfoGainAttributeEval
cmdline.append(f'weka.classifiers.meta.AttributeSelectedClassifier -E \"weka.attributeSelection.InfoGainAttributeEval \" -S \"weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1\" -W weka.classifiers.trees.PBC4cip -- -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth -1 \\\\\\\"-minimalObjByLeaf \\\\\\\" 2 -minimalSplitGain 1.0E-30\\\"\"')

# Attribute selection algorithm - PrincipalComponents
cmdline.append(f'weka.classifiers.meta.AttributeSelectedClassifier -E \"weka.attributeSelection.PrincipalComponents -R 0.95 -A 5\" -S \"weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1\" -W weka.classifiers.trees.PBC4cip -- -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth -1 \\\\\\\"-minimalObjByLeaf \\\\\\\" 2 -minimalSplitGain 1.0E-30\\\"\"')

# Attribute selection algorithm - GainRatioAttributeEval
cmdline.append(f'weka.classifiers.meta.AttributeSelectedClassifier -E \"weka.attributeSelection.GainRatioAttributeEval \" -S \"weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1\" -W weka.classifiers.trees.PBC4cip -- -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees 1000 -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.QuinlanGain \\\\\\\" -maxDepth -1 \\\\\\\"-minimalObjByLeaf \\\\\\\" 2 -minimalSplitGain 1.0E-30\\\"\"')

input_config = cmdline[filename_index]
classifier = from_commandline(input_config, classname="weka.classifiers.Classifier")

print(">>> Building classifier...")
start_time_1 = time.time()
classifier.build_classifier(data)
print(f">>> [Done] Bulding classifier. Time: {(time.time() - start_time_1)} seconds ---")

print(">>> Serializing model...")
start_time_1 = time.time()
classifier.serialize(f"{filename}.model")
print(f">>> [Done] Serializing model. Time: {(time.time() - start_time_1)} seconds ---")

print(">>> Generating big string...")
start_time_1 = time.time()
big_string = str(classifier).split("]")
print(f">>> [Done] Generating big string. Time: {(time.time() - start_time_1)} seconds ---")

list_fields = []
list_supports = []
list_top100 = []
list_top200 = []
list_diff = []
with tqdm(total=len(big_string)) as pbar:
    for item in big_string:
        text = ""
        text = item + "]"
        text = text.split("[")
        fields = text[0]
        # print(fields)
        if (len(text) > 1):
            supports = text[1]
            supports = supports[:-1]
            suuports = supports.split()
            supports_split = supports.split()
            top100 = float(supports_split[0])
            top200 = float(supports_split[1])
            diff = float(top100 - top200)
            list_fields.append(fields.strip())
            list_supports.append(supports_split)
            list_top100.append(top100)
            list_top200.append(top200)
            list_diff.append(diff)
        pbar.update(1)
    
df = pd.DataFrame(columns = ['fields', 'supports', 'top100', 'top200', 'diff'])
df['fields']=list_fields
df['supports']=list_supports
df['top100']=list_top100
df['top200']=list_top200
df['diff']=list_diff
df.to_csv(f"{filename}.csv", index=False)
    

with tqdm(total=len(big_string), file=sys.stdout) as pbar:
    with open(f'D:/GoogleDrive/ITESM/3rd Semester/Tecnicas de ML/Assignment 3/{filename}.txt', 'w') as f:
        print("input: " + input_config, file=f)
        print("output: " + classifier.to_commandline(), file=f)
        print("model:\n", file=f)
        for item in big_string:
            text = item + "]"
            print(text.strip(), file=f)
            pbar.update(1)
    
print(f"--- {(time.time() - start_time)} seconds ---")
    
jvm.stop()