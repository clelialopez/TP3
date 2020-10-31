# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:39:29 2020

@author: Clélia
"""

# =============================================================================
# Librairies
# =============================================================================

 
# Import des librairies
import numpy as np # librairie de calcul numérique
import pandas as pd # librairie de statistiques
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt # librairie de tracé de figures
import matplotlib.patches as mpatches
import time

# Librairies Machine learning
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD # librairie d'analyse factorielle
from sklearn.cluster import KMeans # librairie de clustering
from sklearn.cluster import AgglomerativeClustering # librairie de clustering
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram # librairie de clustering
import collections
from collections import Counter
import sklearn
from scipy.stats.stats import pearsonr  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix #for model evaluation
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


        
#Import
df = pd.read_csv('D:\\creditcard.csv') # chargement de la base de données



