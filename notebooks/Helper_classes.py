class Location_in_Target(object):
    ''' Class to get a list of name of pictures which are labeled with a specific 
        location and the same number of names of pictures which labels do not 
        contain the specific location.
        ...
        Attributes:
        ----------
        location: int 
            number which define a location in a human cell
        pictures: panda.DataFrame
            list of the names of the pictures labeled with the location and 
            same number of pictures without the label
        
        Methods:
        ----------
        save_pictures(pictures, column_name):
            Saves the list of picture names in an Attribute
        get_pictures()
            Returns the list of picture names
        determine_pictures(labels)  
            Find pictures with and without the specific label and call save method
            when list is build
    '''

    def __init__(self, location): 
        '''Parameters
           ----------
           location: int 
                number which define a location in a human cell
        '''
        import pandas as pd
        
        self.location = location
        self.pictures = pd.DataFrame()

    def save_pictures(self, pictures, column_name):
        ''' Saves the list of pictures in the attribute pictures

            Parameters
            ----------
            pictures: panda.DataFrame
                list of the names of the pictures labeled with the location and 
                same number of pictures without the label
            column_name: str
                Description of the data saved in the DataFrame used as column names 
        '''

        self.pictures[column_name] = pictures

    def get_pictures(self):
        '''Gives back the list of picture names which are labeled or not labeled
            with the given location
        '''

        return self.pictures

    def determine_pictures(self, labels, rseed = 23):
        ''' First the names of the pictures labeled with the given location are 
            saved in a list and the method to save it in a DataFrame column is 
            called. 
            The same number of names from pictures not labeled with the location 
            are chosen randomly. The saving function is called with it again. 

            Parameters
            ----------
            labels: DataFrame with all picture names and Targets       
        '''
        import numpy as np
        import random
        
        pictures = [label.Id  for index, label in labels.iterrows() 
                              if str(self.location) in label.Target]
        if len(pictures) > len(labels)/2:
            random.Random(rseed).shuffle(pictures)

            pictures = pictures[:(len(labels)-len(pictures))]
        self.save_pictures(pictures, f'pictures_with_location_{self.location}')
        #The number of pictures labeled with the given location is needed to get the 
        #same number of pictures not labeled with the location
        number_needed_pictures= len(pictures)
        pictures = [label.Id  for index, label in labels.iterrows() 
                              if str(self.location) not in label.Target]
        
        random.Random(rseed).shuffle(pictures)
        pictures_other_label = pictures[:number_needed_pictures]
        self.save_pictures(pictures_other_label,
                                f'pictures_without_location_{self.location}')

class Bin_Embedding(object):
    '''Get the right embeddings for given picture names
        ...
        Attributes:
        ----------
        location: int 
            number which define a location in a human cell
        loc: series
            list of the names of the pictures labeled with the location 
        no_loc: series
            list of the names of the pictures labeled without a given location 
        path: str
            path to the embedded pictures
        df_emb_loc: pandas DataFrame
            Embedded pictures with target
        emb: pandas Dataframe
            all the embedded pictures

        Methods:
        ----------  
        get_embedding():
            gives embedding saves in object and calculate it first, if not done so 
            before
        determine_embedding():
            determine the embedding for given pictures
        get_all_embeddings():
            load all embedded pictures in the attribute 
    '''
    def __init__(self, picture_names, location, path): 
        import pandas as pd
        self.location = location 
        self.loc = picture_names[f'pictures_with_location_{self.location}']
        self.no_loc = picture_names[f'pictures_without_location_{self.location}']
        self.path = path
        self.df_emb_loc = pd.DataFrame()

    def get_embedding(self): 
        ''' If the embedded files are not saved before in the object the determine 
            function is called. The embedded files are given to the caller'''
        if self.df_emb_loc.empty:
            self.get_all_embeddings()
            self.determine_embedding()
        return self.df_emb_loc   

    def determine_embedding(self):
        ''' Build together the embedded pictures for one location and the same amount of
            pictures without the location'''
        import pandas as pd
        from sklearn.utils import shuffle
        # use join to pull out the embeddings of the mitochondria subset
        # first get the mitochondria and set target = 1
        emb_loc = self.emb.merge(self.loc.rename("image_name").to_frame(), how="right", on="image_name")
        emb_loc["target_id"] = 1
        # now the others, target = 0
        emb_no_loc = self.emb.merge(self.no_loc.rename("image_name").to_frame(), how="right", on="image_name")
        emb_no_loc["target_id"] = 0
        # now combine them into one table and shuffle
        self.df_emb_loc = pd.concat([emb_loc, emb_no_loc]).reset_index().drop('index', axis=1)
        self.df_emb_loc = shuffle(self.df_emb_loc, random_state=42)

    def get_all_embeddings(self):
        ''' read all embedded pictures already saved in a given path '''
        import pyarrow.parquet as pq
        # load embeddings (saved as parquet files), convert to pandas and get strings as strings
        self.emb = pq.read_table(self.path).to_pandas()
        for col in ['target_id', 'image_path', 'image_name']:
            self.emb[col] = self.emb[col].str.decode('utf-8')


class Prepared_Test_Train_Data(object):
    ''' Split the data and run through all transformation given by a list of transformation 
        objects. 
        ...
        Attributes:
        ----------
        df_feats: pandas Dataframe
            features - all the embedded pictures without label
        targets: list
            targets - labels
        rseed: int 
            used for stratify the test-train-split
        Methods:
        ----------  
        splitter():
            do the Test_Train_Split
    '''
    def __init__(self, df_emb, rseed = 68):
        import pandas as pd 
        import numpy as np  
        #put the features (embeddings) in one dataframe
        self.df_feats = pd.DataFrame(list(map(np.ravel, df_emb.embedding)))
        # target dataframe
        self.targets = df_emb['target_id']
        self.rseed = rseed

    def splitter(self):
        ''' Split data into train and test set'''
        from sklearn.model_selection import train_test_split
        return train_test_split(self.df_feats, self.targets, random_state= self.rseed, stratify= self.targets)

class Prepare_NN_for_pipline(object):
    ''' Models with neural networks need extra preparation to work in pipelines. This class 
        do the preparations.
        ...
        Attributes:
        ----------
        model: obj
            NN with all layers
        Methods:
        ----------  
        build_layers(self, number_layers, dropout_rate = None, units = 32, kernel_initializer = 'uniform',
                    activation = 'relu', regulizer = 0.000001, input_dim = 1280, activation_last = 'sigmoid' )
            Initialising the NN with regularization and dropout to reduce overfitting
        build_regressor()
            build the regressor for the Pipeline
    '''
    def __init__(self):
        from keras.models import Sequential 
        self.model = Sequential()

    def build_layers(self, number_layers, dropout_rate = None, units = 32, kernel_initializer = 'uniform',
                    activation = 'relu', regulizer = None, input_dim = 1280, activation_last = 'sigmoid' ):
        ''' Initialising the NN with regularization and dropout to reduce overfitting
            Parameters
            ----------
            number_layers: int 
                 number of layers the network shall have
            dropout_rate: float 
                default: None 
                If filled the NN gets a dropout layer after each layer with this as rate
            units: int
                default: 32
                dimensionality of the output space for inner layers
            activation: str
                default: 'relu'
                Activation function to use for first and inner layer
            activation_last: str
                default: 'sigmoid'
                Activation function to use for last layer
            kernel_initializer: str
                default: 'uniform'
                Initializer for the kernel weights matrix.
            regulizer: float 
                default: None or 0.000001
                input for the l2-kernel regularizer
            input_dim: int 
                default: 1280
                size of inputs in NN
            
        '''
        from keras.layers import Dense, Dropout
        from tensorflow.keras import regularizers

        if regulizer != None:
            regulizer = regularizers.l2(regulizer)

        for i in range(number_layers): 
            if i == 0: 
                self.model.add(Dense(units = units, kernel_initializer = kernel_initializer, activation = activation, 
                    kernel_regularizer=regulizer, input_dim = input_dim))
                self.model.add(Dropout(dropout_rate)) 
            elif bool(dropout_rate) and i < number_layers - 1:
                self.model.add(Dense(units = units, kernel_initializer = kernel_initializer, activation = activation, 
                    kernel_regularizer=regulizer))
                self.model.add(Dropout(dropout_rate))  
            else: 
                self.model.add(Dense(units = 1, kernel_initializer = kernel_initializer, activation = activation_last))

    def build_model(self):
        # Compiling the ANN
        self.model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return self.model

    def build_optimizer(self, n_train, batch_size):
        
        import tensorflow as tf
        STEPS_PER_EPOCH = n_train // batch_size

        # hyperbolically decrease the learning rate to 1/2 of base rate after 1000 epochs and so on
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH*1000, # 10
            decay_rate=1,
            staircase=False)
        # Define optimizer used for modelling
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, name='Adam')

    def build_regressor(self, n_train, epochs = 2000, batch_size = 1000, ):
        from keras.wrappers.scikit_learn import KerasRegressor

        self.build_optimizer(n_train, batch_size)

        return KerasRegressor(build_fn=self.build_model, verbose=1, validation_split=0.2, batch_size=batch_size, epochs=epochs)








