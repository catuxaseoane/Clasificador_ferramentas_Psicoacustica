# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import plot_confusion_matrix

import re
import numpy as np
import logging
import timeit
import matplotlib.pyplot as plt

__all__ = [
    'TrainClassifier'
]


class TrainClassifier:
    """
    Class to train a classifier of audio signals
    """

    def __init__(self, X,X1,X2,X3, y):
        self.X = X
        self.X1= X1
        self.X2= X2
        self.X3= X3
        self.y = y

    def train(self):
        """
        Train Random Forest

        :return: pipeline, best_param, best_estimator, perf
        """
        

        logging.info('Splitting train and test set. Test set size: 20%')
        
        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.20,
                                                            random_state=0,
                                                            stratify=self.y)

        X1_train, X1_test, y1_train, y1_test = train_test_split(self.X1, self.y,
                                                            test_size=0.20,
                                                            random_state=0,
                                                            stratify=self.y)

        X2_train, X2_test, y2_train, y2_test = train_test_split(self.X2, self.y,
                                                            test_size=0.20,
                                                            random_state=0,
                                                            stratify=self.y)

        X3_train, X3_test, y3_train, y3_test = train_test_split(self.X3, self.y,
                                                            test_size=0.20,
                                                            random_state=0,
                                                            stratify=self.y)                                                                                                                                                             

        logging.info('Train set size: {0}. Test set size: {1}'.format(y_train.size, y_test.size))

        
       
        pipeline = Pipeline([
            ('scl', StandardScaler()),
            # ('lda', LinearDiscriminantAnalysis()),
            ('clf', SVC(probability=True))
        ])

        # GridSearch
        param_grid = [{'clf__kernel': ['linear', 'rbf'],
                       'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
                       'clf__gamma': np.logspace(-2, 2, 5),
                       # 'lda__n_components': range(2, 17)
                       }]

        estimator = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')


        logging.info('Training model...')
        start = timeit.default_timer()

        model = estimator.fit(X3_train, y3_train)

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        #y_true=
        y_pred = model.predict(X3_test)
        

        perf = {'accuracy': accuracy_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'f1': f1_score(y_test, y_pred, average='macro'),
                # 'summary': classification_report(y_test, y_pred)
                }

        logging.info(perf)

        
        '''
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        titles_options = [("Matriz de confusión non normalizada", None),
                        ("Matriz de confusión normalizada", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(estimator, X_test, y_test,
                                        display_labels=None,
                                        cmap=plt.cm.Blues,
                                        normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

        plt.show()
        '''

        ##################################################################GRÁFICAS DE DISPERSIÓN
        font = {'size': 22 } 
        font2 = {'size': 18 } 
        '''for x in range(0, y_test.size):
            #Ferramenta boa
            if(y_test[x].find("GOOD")!=-1):
                fgood=plt.scatter(x=(X2_test[x,:].tolist(),X2_test[x,:]),y=(X2_test[x,:].tolist(),X2_test[x,:]), marker='x',color='#1eb53a')

            #Ferramenta mala
            if(y_test[x].find("BAD")!=-1):
                fbad=plt.scatter(x=(X2_test[x,:].tolist(),X2_test[x,:]),y=(X2_test[x,:].tolist(),X2_test[x,:]), marker='x',color='#e8112d')


        fgood=plt.scatter([],[],marker='x',color='#1eb53a')
        fbad=plt.scatter([],[],marker='x',color='#e8112d')
        plt.legend((fgood,fbad),
        ('Ferramenta en bo estado', 'Ferramenta en mal estado'),
        scatterpoints=1,
        loc='lower right',
        ncol=3,
        fontsize=15)
        plt.xlabel('Sharpness [acum]',fontdict=font2)
        plt.ylabel('Sharpness [acum]',fontdict=font2)
        plt.title("Diagrama de dispersión - Sharpness",fontdict=font)
        plt.show()
        '''
        
        '''for x in range(0, y_test.size):
            
            if (y_test[x].find("GOOD")!=-1):
                plt.scatter(x=(X2_test[x,:].tolist(),X2_test[x,:]),y=(X3_test[x,:].tolist(),X3_test[x,:]), marker='x',color='#1eb53a')
            
                
            if (y_test[x].find("BAD")!=-1):
                fbad=plt.scatter(x=(X2_test[x,:].tolist(),X2_test[x,:]),y=(X3_test[x,:].tolist(),X3_test[x,:]), marker='x',color='#e8112d')
                
        
        fgood=plt.scatter([],[], marker='x',color='#1eb53a')
        fbad=plt.scatter([],[], marker='x',color='#e8112d')    
        plt.legend((fgood,fbad),
           ('Ferramenta en bo estado', 'Ferramenta en mal estado'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=15)
        
        plt.xlabel('Característica 1 = Sharpness [acum]',fontdict=font2) 
        plt.ylabel('Característica 2 = Roughness [asper]',fontdict=font2) 
        plt.title("Diagrama de dispersión - Sharpness e Roughness",fontdict=font)
            
        plt.show()
        '''
        #plt.scatter(X_test.tolist(),X_test)
        #plt.show()

       

        


        

        return perf, model.best_params_, model.best_estimator_



