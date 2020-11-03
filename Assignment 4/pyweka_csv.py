import sys
import os
import weka.core.jvm as jvm
# from weka.core.converters import Loader
from weka.core import converters
from weka.classifiers import Classifier
import weka.plot.graph as graph  # NB: pygraphviz and PIL are required
from weka.core.classes import Random, from_commandline
import weka.core.serialization as serialization
from weka.filters import Filter
import re
from tqdm import tqdm
import time
import pandas as pd

df_importante = pd.read_csv("dataset_ready.csv", encoding='utf8')
end_name = 'csv_500'

if not os.path.exists(end_name):
    try:
        os.makedirs(end_name)
    except OSError:
        print ("Creation of the directory %s failed. Try again." % end_name)
        jvm.stop()
    else:
        print ("Successfully created the directory %s " % end_name)

lista_lda = []
for index, row in enumerate(df_importante.itertuples(), 1):
    topic = row.lda_topic
    if topic == 1:
        lista_lda.append('topic_1')
    elif topic == 2:
        lista_lda.append('topic_2')
    elif topic == 3:
        lista_lda.append('topic_3')
    elif topic == 4:
        lista_lda.append('topic_4')
    elif topic == 5:
        lista_lda.append('topic_5')
        
df_importante['lda_topic'] = lista_lda

df_arr = []
list_file_names = []
for col in df_importante.iloc[:, 0:11].columns:
    temp = pd.get_dummies(df_importante[col])
    for x in range(len(temp.columns)):
        temp_2 = df_importante.copy()
        temp_2['class'] = temp.iloc[:, x]
        temp_2['class'] = ["1" if x == 1 else "2" for x in temp_2['class']]
        del temp_2[col]
        temp_2.to_csv(f"{end_name}/{str(col)}_{str(temp.columns[x])}.csv", index=False, encoding='cp1252')
        df_arr.append(pd.DataFrame(temp_2))
        list_file_names.append(f"{str(col)}_{str(temp.columns[x])}")
        
start_time = time.time()

jvm.start(packages=True, max_heap_size='12g')

print("Welcome")

print("\n\n")
print(">>> Start...\n\n")

data_dir = f"{end_name}/"

# Criterions
numTrees = 500
maxDepth = 10
minObjects = 31
.

for item in list_file_names:
    print(f"Analizing: [{item}]")
    file_name = str(item)
    # file_name = 'iris'
    file_extension = ".csv"
    csv_file = file_name + file_extension

    path = f'results_{end_name}\\{file_name}'
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed. Try again." % path)
            jvm.stop()
        else:
            print ("Successfully created the directory %s " % path)

    print(f"Loading: {data_dir + csv_file}")
    loader = converters.Loader(classname='weka.core.converters.CSVLoader')
    print(f"Loader: {loader}")
    data = loader.load_file(data_dir + csv_file)
    print(f"[Done] Loading: {data_dir + csv_file}")
    
    # remove class attribute
    print(f"Numeric to nominals")
    remove = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "8,11,12,13,14,25"])
    remove.inputformat(data)
    filtered = remove.filter(data)
    filtered.class_is_last()
    print(f"[Done] Numeric to nominals")

    attributes = (filtered.attribute_stats(filtered.num_attributes - 1))
    num_class1 = (attributes.nominal_counts[0])
    num_class2 = (attributes.nominal_counts[1])
    
    input_config = f'weka.classifiers.trees.PBC4cip -S 1 -miner \"PRFramework.Core.SupervisedClassifiers.EmergingPatterns.Miners.RandomForestMinerWithoutFiltering -bagSizePercent 100 -numFeatures -1 -numTrees {numTrees} -builder \\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.Builder.DecisionTreeBuilder -distributionEvaluator \\\\\\\"PRFramework.Core.SupervisedClassifiers.DecisionTrees.DistributionEvaluators.Hellinger \\\\\\\" -maxDepth {maxDepth} \\\\\\\"-minimalObjByLeaf \\\\\\\" {minObjects} -minimalSplitGain 1.0E-30\\\"\"'

    classifier = from_commandline(input_config, classname="weka.classifiers.Classifier")

    print(">>> Building classifier...")
    start_time_1 = time.time()
    classifier.build_classifier(filtered)
    print(f">>> [Done] Bulding classifier. Time: {(time.time() - start_time_1)} seconds ---")

    print(">>> Serializing model...")
    start_time_1 = time.time()
    classifier.serialize(f"{path}/{file_name}.model")
    print(f">>> [Done] Serializing model. Time: {(time.time() - start_time_1)} seconds ---")

    print(">>> Generating big string...")
    start_time_1 = time.time()
    big_string = str(classifier).split("]")
    print(f">>> [Done] Generating big string. Time: {(time.time() - start_time_1)} seconds ---")

    list_num_class1 = []
    list_num_class2 = []
    list_fields = []
    list_supports = []
    list_class1 = []
    list_class2 = []
    list_c_total = []
    list_diff = []
    list_class1_count = []
    list_class2_count = []
    list_diff_c_pattern = []
    list_div_c_pattern = []
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
                
                class1 = float(supports_split[0])
                class2 = float(supports_split[1])
                class1_count = class1 * num_class1
                class2_count = class2 * num_class2
                
                if (class1_count + class2_count) > 0:
                    division = (class1_count) / ((class1_count) + (class2_count))
                else:
                    division = -1
                diff = float(class1 - class2)
                
                list_num_class1.append(num_class1)
                list_num_class2.append(num_class2)
                list_c_total.append(num_class1 - num_class2)
                list_fields.append(fields.strip())
                list_supports.append(supports_split)
                list_class1.append(class1)
                list_class2.append(class2)
                list_diff.append(diff)
                list_class1_count.append(class1_count)
                list_class2_count.append(class2_count)
                list_diff_c_pattern.append(class1_count - class2_count)
                list_div_c_pattern.append(division)
            pbar.update(1)
        
    df = pd.DataFrame()
    df['pattern'] = list_fields
    df['pattern_items'] = df.pattern.str.count("AND") + 1
    df['supports'] = list_supports
    df['s1'] = list_class1
    df['s2'] = list_class2
    df['diff_s'] = list_diff
    df['c1_total'] = list_num_class1
    df['c2_total'] = list_num_class2
    df['diff_c_total'] = list_c_total
    df['c1_pattern'] =  list_class1_count
    df['c2_pattern'] =  list_class2_count
    df['diff_c_pattern'] = list_diff_c_pattern
    df['div_c_pattern'] = list_div_c_pattern
    
    df = df.drop_duplicates(subset=['pattern'])
    
    df2 = pd.DataFrame(df[(df['diff_s'] > 0.3) & (df['diff_c_pattern'] > 30) & (df['div_c_pattern'] > 0.6)])
    
    df.to_csv(f"{path}/{file_name}_full.csv", index=False)
    df2.to_csv(f"{path}/{file_name}_restrictions_applied.csv", index=False)
  
    with tqdm(total=len(big_string), file=sys.stdout) as pbar:
        with open(f'{path}/{file_name}.txt', 'w') as f:
            print("input: " + input_config, file=f)
            print("output: " + classifier.to_commandline(), file=f)
            print("model:\n", file=f)
            for item in big_string:
                text = item + "]"
                print(text.strip(), file=f)
                pbar.update(1)
        
    print(f"--- {(time.time() - start_time)} seconds ---")
    print(f"Done analizing: [{item}]\n\n")
    
print("FULL DONE")
    
jvm.stop()