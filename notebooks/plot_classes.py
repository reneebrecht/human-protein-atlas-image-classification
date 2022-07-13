class OrderLabels(object):
    ''' ---
        ...
        Attributes:
        ----------
        label_names: dict 
                all possible locations in the cell
        target_labels: pd.DataFrame 
                targets (locations) for each picture name
        upper_ylim: int
                adaptable to the number of counts for better visualization
        df_order: pd.DataFrame
                data frame with ordered labels
        
        
        Methods:
        ----------
        ordered_dataframe(pictures, column_name):
            Saves the list of picture names in an Attribute
        plot_ordered_labels():
            Plot an ordered data frame
        plot_ordered_labels_with_names()
            Plot an ordered data frame and show the location names
        
    '''


    def __init__(self, target_labels, upper_ylim=15000): 
        '''Parameters
           ----------
           target_labels: pd.DataFrame 
                targets for each picture name
            upper_ylim: int
                adaptable to the number of counts for better visualization
        '''
        self.target_labels = target_labels
        self.upper_ylim =upper_ylim
        self.label_names = {
            0:  "Nucleoplasm",  
            1:  "Nuclear membrane",   
            2:  "Nucleoli",   
            3:  "Nucleoli fibrillar center",   
            4:  "Nuclear speckles",
            5:  "Nuclear bodies",   
            6:  "Endoplasmic reticulum",   
            7:  "Golgi apparatus",   
            8:  "Peroxisomes",   
            9:  "Endosomes",   
            10:  "Lysosomes",   
            11:  "Intermediate filaments",   
            12:  "Actin filaments",   
            13:  "Focal adhesion sites",   
            14:  "Microtubules",   
            15:  "Microtubule ends",   
            16:  "Cytokinetic bridge",   
            17:  "Mitotic spindle",   
            18:  "Microtubule organizing center",   
            19:  "Centrosome",   
            20:  "Lipid droplets",   
            21:  "Plasma membrane",   
            22:  "Cell junctions",   
            23:  "Mitochondria",   
            24:  "Aggresome",   
            25:  "Cytosol",   
            26:  "Cytoplasmic bodies",   
            27:  "Rods & rings"
        }


    def ordered_dataframe(self):
        '''Create an ordered data frame
        '''
        import pandas as pd
        from itertools import chain 
        from collections import Counter

        self.target_labels['target_list'] = self.target_labels['Target'].map(lambda x: [int(a) for a in x.split(' ')])
        count_labels = Counter(list(chain.from_iterable(self.target_labels['target_list'].values)))
        df_count_labels = pd.DataFrame.from_dict(self.label_names, orient='index').rename_axis('key')
        df_count_labels.columns = ['Loc']
        df_count_labels['count'] = df_count_labels.index.map(count_labels)
        df_count_labels = df_count_labels.sort_values('count', ascending=False).reset_index()
        self.df_order = df_count_labels.rename_axis('occurence_order').reset_index()

        return self.df_order

    def plot_ordered_labels(self):
        '''Plot an ordered data frame
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(figsize=(21,3))

        sns.barplot(x='occurence_order', y='count', data=self.df_order, color='#36cf43')
        plt.ylim(0,self.upper_ylim)
        plt.xticks(self.df_order['occurence_order'])
        xlab = self.df_order['key'].astype(str).to_list()
        ax.set_xticklabels(xlab, fontsize=18)

        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')
        plt.xlabel('protein localization', fontsize=18)

        for i in self.df_order['occurence_order']:
            plt.text(i-0.4,self.df_order['count'].iloc[i], self.df_order['count'].iloc[i], size=16)

        sns.despine(left=True, bottom=True)

    def plot_ordered_labels_with_names(self):
        '''Plot an ordered data frame and show the location names
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(figsize=(4,9))

        sns.barplot(x='count', y='Loc', data=self.df_order, color='#36cf43')

        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([])
        for p in ax.patches:
            ax.annotate("%.2f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y() + 0.9),
                        xytext=(5, 10), textcoords='offset points')

        sns.despine(left=True, bottom=True);

class ConfusionErrorMetrics(object):
    ''' ---
        ...
        Attributes:
        ----------
        df_order: pd.DataFrame
            data frame with ordered labels
        tn: int
            true negatives
        tp: int
            true positives
        fn: int
            false negatives
        fp: int
            false positives
        df_metrics: pd.DataFrame
            data frame with confusion instances and error metrics
        df_metrics_o: pd.DataFrame
            data frame with confusion instances and error metrics ordered by the occurence of the target

        
        
        
        Methods:
        ----------
        calculate_error_metrics()
            Create df_metrics data frame with confusion and error metrics
        metrics_names_and_order()
            Append data frame df_metrics with label names and occurrence order
            
    '''
    
    def __init__(self, df_order, y_test=None, y_pred=None, TP=None, FN=None, FP=None, TN=None): 
        '''Parameters
           ----------
            y_test: pd.Series
                Part of the target for testing the model
            y_pred: pd.Series
                Outcome of the models prediction
            df_order: pd.DataFrame
                data frame with ordered labels (class OrderLabels)
                should contain column 'key' from label_name dictionary
            TP, FN, FP, TN: int or array
                4 optional confusion instances (use all or none)
        '''
        self.df_order = df_order

        from sklearn.metrics import multilabel_confusion_matrix

        try:
            if not TP.empty:
                self.tp=TP
                self.fn=FN
                self.fp=FP
                self.tn=TN

        except AttributeError:
            mcm = multilabel_confusion_matrix(y_test, y_pred)

            self.tn = mcm[:, 0, 0]
            self.tp = mcm[:, 1, 1]
            self.fn = mcm[:, 1, 0]
            self.fp = mcm[:, 0, 1]

    def calculate_error_metrics(self):
        '''Create a data frame with confusion and error metrics
        '''
        import pandas as pd

        recall = self.tp / (self.tp + self.fn)
        precision = self.tp/ (self.fp+self.tp)
        specificity = self.tn / (self.tn + self.fp)
        f1 = (2*self.tp)/(2 * self.tp + self.fp + self.fn)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fn + self.fp)

        self.df_metrics = pd.DataFrame({'recall': recall, 'precision': precision, 'f1':f1, 'accuracy':accuracy, 
                            'TP':self.tp, 'FP':self.fp, 'FN':self.fn, 'TN':self.tn})
        
    
    def metrics_names_and_order(self):
        '''Append existing data frame df_metrics with label names and occurrence order
        '''
        import pandas as pd

        self.calculate_error_metrics()

        self.df_metrics_o = pd.merge(self.df_metrics, self.df_order, left_index=True, right_on='key')
        return self.df_metrics_o


class MulticlassPlots(object):
    ''' ---
        ...
        Attributes:
        ----------
        df_metrics_o: pd.DataFrame
            data frame with confusion instances and error metrics ordered by the occurence of the target

        self.df_metrics_s: pd.DataFrame
            df_metrics with stacked TP, FN, FP on occurence_order for grouped plotting
        
        
        Methods:
        ----------
        plot_f1_score()
            Plots the f1-score for multiple targets
        plot_all_error_metrics()
            Returns the list of picture names
        stack_dataframe()
            Stack data frame with stacked TP, FN, FP on occurence_order in preparation for plotting confusion instances
        plot_confusion_instances()
            Plot confusion instances TP, FN, FP
        plot_share_of_positives(self):
            Plot true positives (TP) as a share of all positives (TP + FN) 
        plot_f1_score_with_train(self, f1_score_train):
            Plot F1 score for all labels with F1_score for training
    '''
    
    def __init__(self, df_metrics_o): 
        '''Parameters
           ----------
           df_metrics_o: pd.Dataframe
            dataframe with error metrics and order of occurence
        '''
        self.df_metrics_o = df_metrics_o   

    def plot_f1_score(self):
        '''Plot F1 score for all labels
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(figsize=(21,3))

        sns.set(style="ticks")
        sns.axes_style("ticks")

        sns.scatterplot(x='occurence_order', y='f1', data=self.df_metrics_o, color='#4C9A2A', s=500, label='F1-score')
        
        plt.ylim(-0.1,1.1)
        plt.xticks(self.df_metrics_o['occurence_order'])
        plt.xlabel('protein localization', fontsize=18)
        plt.ylabel(' bad         good', fontsize=18)

        xlab = self.df_metrics_o['key'].astype(str).to_list()
        ax.set_xticklabels(xlab, fontsize=18)
        ax.xaxis.grid(True)
        ax.tick_params(axis='y', labelsize=18)
        ax.get_legend().remove()

        ynew = 0.5
        ax.axhline(ynew, linestyle=':', color='grey', label='magic line')
        sns.despine(left=True, bottom=True);


    def plot_all_error_metrics(self):
        '''Plot F1 score, precision, recall, accuracy for all labels 
            and a threshold line
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(figsize=(21,3))

        sns.set(style="ticks")
        sns.axes_style("ticks")

        sns.scatterplot(x='occurence_order', y='f1', data=self.df_metrics_o, color='#4C9A2A', s=500, label='F1-score')
        sns.scatterplot(x='occurence_order', y='precision', data=self.df_metrics_o, color='#c9851f', s=500, alpha=0.2, label='precision')
        sns.scatterplot(x='occurence_order', y='recall', data=self.df_metrics_o, color='#776280', s=500, alpha=0.2, label='recall')
        sns.scatterplot(x='occurence_order', y='accuracy', data=self.df_metrics_o, color='#95382b', s=500, alpha=0.2, label='accuracy')

        plt.ylim(-0.1,1.1)
        plt.xticks(self.df_metrics_o['occurence_order'])
        plt.xlabel('protein localization', fontsize=18)
        plt.ylabel(' bad         good', fontsize=18)

        xlab = self.df_metrics_o['key'].astype(str).to_list()
        ax.set_xticklabels(xlab, fontsize=18)
        ax.xaxis.grid(True)
        ax.tick_params(axis='y', labelsize=18)

        ynew = 0.5
        ax.axhline(ynew, linestyle=':', color='grey', label='magic line')

        plt.legend(fontsize=14);

    def stack_dataframe(self):
        '''Stack data frame in preparation for plotting confusion instances
        '''
       
        df_plot_counted = self.df_metrics_o[['TP', 'FN', 'FP']].copy()
        self.df_metrics_s = df_plot_counted.stack().reset_index().rename(columns={'level_0': 'value_index','level_1': 'instance', 0: 'value'})
        
        


    def plot_confusion_instances(self):
        '''Plot confusion instances TP, FN, FP
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        self.stack_dataframe()

        palette = ['#1e7507', '#f9ac07', '#c77406']
        sns.set_palette(sns.color_palette(palette))

        fig,ax = plt.subplots(figsize=(21,3))

        sns.barplot(x='value_index', y='value', data=self.df_metrics_s, hue='instance')

        plt.xticks(self.df_metrics_o['occurence_order']) ###
        xlab = self.df_metrics_o['key'].astype(str).to_list()
        ax.set_xticklabels(xlab, fontsize=18)
        plt.xlabel('protein localization', fontsize=18)
        plt.ylabel('confusion value', fontsize=18)

        ax.xaxis.grid(True)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)

        plt.legend(fontsize=14);  

    def plot_share_of_positives(self):
        '''Plot true positives (TP) as a share of all positives (TP + FN)
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt   

        self.df_metrics_o['share_TP'] = (100 /(self.df_metrics_o['TP'] + self.df_metrics_o['FN'])) * self.df_metrics_o['TP']
        
        fig,ax = plt.subplots(figsize=(21,3))

        sns.barplot(x='occurence_order', y='share_TP', data=self.df_metrics_o, color='#bb6678')
        
        plt.xticks(self.df_metrics_o['occurence_order'])
        plt.ylim(0,110)
        plt.xlabel('protein localization', fontsize=14)
        plt.ylabel('share of true positives %', fontsize=14)
        
        xlab = self.df_metrics_o['key'].astype(str).to_list()
        ax.set_xticklabels(xlab, fontsize=18)
        ax.xaxis.grid(True)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)

        ynew = 50
        ax.axhline(ynew, linestyle=':', color='grey', label='50 %')
        plt.legend(fontsize=14);

    def plot_f1_score_with_train(self, f1_score_train):
        '''Plot F1 score for all labels with F1_score for training
        '''
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        #put f1_score for data in table
        self.df_metrics_o['f1_score_train'] = f1_score_train

        fig,ax = plt.subplots(figsize=(21,3))

        sns.set(style="ticks")
        sns.axes_style("ticks")

        sns.scatterplot(x='occurence_order', y='f1', data=self.df_metrics_o, color='#4C9A2A', s=500, label='F1-score')
        sns.scatterplot(x='occurence_order', y='f1_score_train', data=self.df_metrics_o, color='#4C9A2A', s=500, label='F1-score', alpha = 0.2)


        plt.ylim(-0.1,1.1)
        plt.xticks(self.df_metrics_o['occurence_order'])
        plt.xlabel('protein localization', fontsize=18)
        plt.ylabel(' bad         good', fontsize=18)

        xlab = self.df_metrics_o['key'].astype(str).to_list()
        ax.set_xticklabels(xlab, fontsize=18)
        ax.xaxis.grid(True)
        ax.tick_params(axis='y', labelsize=18)
        ax.get_legend().remove()

        ynew = 0.5
        ax.axhline(ynew, linestyle=':', color='grey', label='magic line')
        sns.despine(left=True, bottom=True);
