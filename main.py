import os
import pickle

import numpy as np
import pandas as pd
import shap
from flask import Flask, render_template, redirect, send_file, request
from numpy import mean
from pydriller import Repository
from scipy.stats import entropy


def extract_feature_functionality(class_name, package_name=None):
    res = {}
    items = {
         'controller':"manage, process, control, ctrl, command, cmd, process, proc, ui, drive, system, subsystem, parser, service".split(', '),
         'procedural':"make, create, factory, exec, compute, display, view, calculate, batch, thread, cluster".split(', '),
         'test':['test','junit'],
         'util':['util','helper'],
     }

    if package_name is None:
        items['external'] = []
    else:
        items['external'] = ['org.apache.cayenne']
    for k,v in items.items():
        res[f'is_{k}'] = False
    for k,v in items.items():
        if not res[f'is_{k}']:
            if k != 'external':
                for vv in v:
                    if vv in class_name.lower():
                        res[f'is_{k}'] = True
                        break

            else:
                for vv in v:
                    if vv not in class_name.lower():
                        res[f'is_{k}'] = True
                        break
    return res

def dsc(c1,c2):
    arr1 = c1.split('\\')
    arr2 = c2.split('\\')
    for i,elem1 in enumerate(arr1):
        if elem1 != arr2[i]:
            break
    return len(arr1)+len(arr2)-2*i

def dsc_sets(s):
    s_java = []
    for i in s:
        if i is None: continue
        if '.java' in i.lower():continue
        s_java.append(i)
    res = []
    if len(s_java) > 1:
        for ii in s_java:
            for jj in s_java:
                res.append(dsc(ii,jj))
    else:
        return 0
    return mean(res) if len(res)>0 else 0

def extract_feature(dir_prj, filepath, commit=None):
    print(filepath)
    i = 0
    files = set()
    files_list = []
    modified_file_numbers = []
    res = {
        'churn':0,
        'files':0,
        'NR':0,
        'NBF':0,
    }
    author_name_map = {}
    contributions = []
    total_commits = 0
    repo = Repository(dir_prj, to_commit=commit, filepath=filepath)
    if commit is None:
        repo = Repository(dir_prj, filepath=filepath)
    for commit in repo.traverse_commits():
        if commit.files > 100:
            continue
        i+=1
        res['churn'] += commit.lines
        res['files'] += commit.files
        files = files.union(ff.new_path for ff in commit.modified_files)
        files_list.extend(commit.modified_files)
        modified_file_numbers.append(len(commit.modified_files))
        # res['ce'] += entropy(CE_map[class_name])
        if 'refactor' in commit.msg.lower() or 'restruct' in commit.msg.lower():
            res['NR'] += 1
        if 'fix' in commit.msg.lower() or 'bug' in commit.msg.lower() or ('[' in commit.msg.lower() and ']' in commit.msg.lower()):
            res['NBF'] += 1

        total_commits += 1
        name = commit.author.name
        if author_name_map.get(name) is None:
            author_name_map[name] = 1
        else:
            author_name_map[name] += 1
    for k_author, v_author in author_name_map.items():
        contributions.append(v_author)


    if i == 0:
        print(filepath+' missing')
        return res
    res['nc'] = i
    res['ncom'] = len(contributions)
    res['avg_cs'] = res['files'] / i
    res['dsc'] = dsc_sets(files)
    res['ce'] = entropy(modified_file_numbers)
    res['exp'] = mean(contributions)

    res['own'] = max(contributions) / sum(contributions) if total_commits != 0 else 1

    return res



def extract(x_test, idx, features):
    with open('./model.pkl','rb') as f1:
        cls = pickle.load(f1)
    y_predict = cls.predict(x_test)

    explainer = shap.TreeExplainer(cls)
    shap_values = explainer(x_test)
    classes = int(y_predict[0])

    row = 0
    # shap will throw an error when displaying boolean values, source code should be modified to fix it
    #  the statement
    #                 yticklabels[rng[i]] = features[order[i]] + " = " + feature_names[order[i]]
    #  should be changed to:
    #                 yticklabels[rng[i]] = str(features[order[i]]) + " = " + str(feature_names[order[i]])
    pic = f'/img/{idx}_{classes}.png'
    if not os.access(pic,os.F_OK):
        plt = shap.waterfall_plot(shap.Explanation(values=
                                                   shap_values.values[:, :, classes][row],
                                                   base_values=explainer.expected_value[classes],
                                                   data=x_test[0, :],
                                                   feature_names=features),
                                  max_display=10, show=False)

        plt.savefig(f'./result/{idx}_{classes}.png', bbox_inches='tight')
        plt.clear()
    return pic, y_predict

def generate_features(path):
    file = f'./res_{path}.csv'
    jar_name = './lib/ck-0.7.1-SNAPSHOT-jar-with-dependencies.jar'
    path = './demo/'+path
    if os.access(file, os.F_OK):
        pass
    else:
        ck_jar = f'java -jar {jar_name} {path} false 0 true ./tmp/'
        os.system(ck_jar)
        metrics_table = pd.read_csv('./tmp/class.csv')
        metrics_table = metrics_table[metrics_table['type'].isin(['class'])]
        res_df = pd.DataFrame()
        for idx, row in metrics_table.iterrows():
            file = row['file']
            process = extract_feature(path, file)
            for k,v in process.items():
                row[k]=v
            functionality = extract_feature_functionality(row['class'])
            for k,v in functionality.items():
                row[k]=v
            res_df = res_df._append(row, ignore_index=True)
        res_df.to_csv(file,index=False)
    return pd.read_csv(file).replace(np.nan,0).replace(-1,0)

metrics = ['is_controller', 'lambdasQty', 'privateFieldsQty', 'defaultFieldsQty', 'innerClassesQty', 'staticMethodsQty', 'loopQty', 'stringLiteralsQty', 'NBF', 'abstractMethodsQty', 'dit', 'logStatementsQty', 'uniqueWordsQty', 'defaultMethodsQty', 'is_procedural', 'own', 'churn', 'lcom', 'fanin', 'protectedMethodsQty', 'modifiers', 'protectedFieldsQty', 'returnQty', 'finalFieldsQty', 'anonymousClassesQty', 'tryCatchQty', 'dsc', 'synchronizedFieldsQty', 'synchronizedMethodsQty', 'exp', 'NR', 'privateMethodsQty', 'noc', 'publicFieldsQty', 'lcom*', 'fanout', 'finalMethodsQty', 'parenthesizedExpsQty', 'is_util', 'avg_cs', 'tcc', 'nosi', 'is_test']
app = Flask(__name__)


@app.route("/",methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        os.system(f'cd ./demo && git clone {url}')
        return gen_features(url.split('/')[-1])
    else:
        return render_template('index.html')

@app.route("/img/<string:pic>")
def png(pic):
    return send_file(f'./result/{pic}', mimetype='image/png')

@app.route("/project/<string:path>/")
def gen_features(path):
    metrics_table = generate_features(path)
    return redirect(f'/project/{path}/0')

@app.route("/project/<string:path>/all")
def detail_all(path):
    path = f'./demo/{path}'
    metrics_table = generate_features(path)
    available_classes = list(metrics_table['class'])
    for i in range(0, len(available_classes)):
        row = metrics_table.iloc[i]
        pic, y_predict = extract(np.array(row[metrics]).reshape(-1, len(metrics)), i, metrics)

    return display(0, metrics_table, available_classes)

@app.route("/project/<string:path>/<int:idx>")
def detail(path, idx):
    metrics_table = generate_features(path)
    available_classes = list(metrics_table['class'])
    if len(available_classes) > 0:
        return display(idx, metrics_table, available_classes)
    else:
        return render_template("detail.html", available_classes=[])

def display(idx, metrics_table, available_classes):
    row = metrics_table.iloc[idx]
    class_name = row['class']
    pic, y_predict = extract(np.array(row[metrics]).reshape(-1, len(metrics)), idx, metrics)
    res_arr = ['Low','Mid','High']
    try:
        with open(row['file'],'r') as f1:
            code = f1.read()
    except:
        code = ''
    return render_template("detail.html",
                           available_classes=available_classes,
                           pic=pic,
                           y_predict=res_arr[y_predict[0]],
                           class_name=class_name,
                           code=code,
                           idx=idx)
