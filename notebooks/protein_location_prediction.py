class predict_protein_location(object):
    ''' Class to predict the location(s) of proteins in a given image set (3 colors)
        ...
        Attributes:
        ----------
        image_dir: str 
            path to folder containing image files (ends with '/')
        filename: str
            the base of the image filenames, i.e., without e.g. '_green.png'
        model_dir: str
            path to the folder containing the pre-trained binary models (ends with '/')

        Methods:
        ----------
        run_prediction():
            run all methods in the correct order
        read_and_combine_files():
            read and combine image files and return 3 layer array as attribute
        get_embeddings()
            run prediction with pre-trained model and return embeddings as attribute
        get_location_names()
            add the protein location names as an attribute
        load_binary_models(labels)  
            load the binary models and return a list of them as attribute
        predict_w_binary_models()
            return binary model predictions as an attribute
    '''

    def __init__(self, image_dir, filename, model_dir, min_probability_threshold=1): 
        '''Parameters
           ----------
           location: int 
                number which define a location in a human cell
        '''
        from efficientnet.tfkeras import EfficientNetB0
        #import tensorflow as tf

        self.image_dir = image_dir
        self.filename = filename
        self.path_filename = image_dir + filename
        self.image_colors_to_include = ['_red.png', '_blue.png', '_green.png'] # colors to be used in analysis
        self.model_dir = model_dir+'*'
        self.min_probability_threshold = min_probability_threshold
        self.embedding_model = EfficientNetB0(weights='imagenet', include_top=False, pooling="avg")
        #self.embedding_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, pooling="avg")
        self.get_location_names()

    def run_prediction(self):
        self.combine_images()
        self.get_embeddings()
        self.load_binary_models()
        self.predict_w_binary_models()

    def get_location_names(self):
        """create a list of the name and number of each protein location
        Args:
        Returns:
            label_names: a dictionary of protein location numbers and names
        """
        # courtesy of https://www.kaggle.com/code/allunia/protein-atlas-exploration-and-baseline
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
        
    def load_images(self,end_str):
        """reading in a png image file and decoding
        Args:
            end_str(str): end of the filename e.g. ['_red.png', '_blue.png', '_green.png'] 
        Returns:
            array of the image
        """
        import tensorflow as tf
        return tf.io.decode_png(tf.io.read_file(self.path_filename+end_str))

    def combine_images(self):
        """combine 3 image files into one array, convert to float32 and expand dims for model input
        Returns:
            im_array: array containing all 3 images as different layers
        """
        import tensorflow as tf
        im_array = tf.concat([self.load_images(end_str) for end_str in self.image_colors_to_include], axis=2)        
        im_array = tf.image.convert_image_dtype(im_array, tf.float32)
        if len(im_array.shape) < 4: im_array = tf.expand_dims(im_array, axis=0)
        self.im_array = im_array

    def get_embeddings(self):
        """run prediction with pre-trained model (EfficientNetB0) to get embeddings
        Returns:
            embeddings: a vector of numbers, which is the embeddings of the image
        """
        self.embeddings = self.embedding_model.predict(self.im_array, verbose=1)

    def load_binary_models(self):
        """loading the pre-trained binary model for each protein location, and its
        corresponding location number
        Returns:
            models: list containing all the pre-trained models
            model_location: list of the location corresponding to each model
        """
        import joblib, glob
        self.models = [joblib.load(f)[-1] for f in glob.glob(self.model_dir)] 
        # get the corresponding location number of each model
        self.model_location = [f.split('_')[-1] for f in glob.glob(self.model_dir)] 

    def predict_w_binary_models(self):
        """predicting the protein location of an image, using embeddings, and return all locations
        with a probability >= min_probability_threshold
        Returns:
            predictions_df: a dataframe containing the protein location name and number and the probability that a protein
            is in that location
            predicted_locations: list of all predicted protein locations for this image (prob >= min_probability_threshold), 
            sorted by probability
        """
        import pandas as pd
        # create a dataframe containing location name, number and probability that it is in the image
        self.predictions_df =  pd.DataFrame({'protein_location': [int(i) for i in self.model_location],
                                            'prediction_probability': [f.predict_proba(self.embeddings)[0][1] for f in self.models] })
        self.predictions_df = self.predictions_df.sort_values('protein_location').reset_index(drop=True) # sort to match location names
        self.predictions_df.insert(0, 'protein_location_names', self.label_names.values()) # add location name as first column
        # get the predicted locations
        self.predicted_locations = (self.predictions_df.query('prediction_probability >= @self.min_probability_threshold').
                                    sort_values('prediction_probability',ascending=False).protein_location.ravel() )
    

    def plot_predicted_images(self, train_labels, axs, fcolor='black'):
        import matplotlib.pyplot as plt
        #%matplotlib inline
        import numpy as np

        axs.imshow(self.im_array.numpy().squeeze(), aspect='equal')
        axs.set_xticks([])
        axs.set_yticks([])
        # getting TP, FN, FP
        Target_list = np.array([int(x) for x in train_labels.query('Id == @self.filename').Target.values[0].split(' ')])        
        Predict_list = self.predicted_locations
        intersect_np = np.intersect1d(Target_list, Predict_list)
        intersect_str = ', '.join(str(x) for x in intersect_np)
        missed_str = ', '.join(str(x) for x in Target_list[~np.isin(Target_list,intersect_np)])
        incorrect_str = ', '.join(str(x) for x in Predict_list[~np.isin(Predict_list,intersect_np)])
        axs.set_title(label='Correct: '+ intersect_str, size=16, color=fcolor)
        #axs.set_xlabel(r'Missed: '+missed_str+'\n'+ r' Incorrect: '+incorrect_str, size=16)
        axs.set_xlabel(r"Missed: {0} " "\n" r"Incorrect : {1}".format(missed_str, incorrect_str), color=fcolor, size=14 )
        # axs.text(0., 0.1, 'Target: '+ str(Target_list), size=12, color='white', horizontalalignment='left',
        #             verticalalignment='top', transform=axs.transAxes)
        # axs.text(0., 0.95, 'Predict: '+ str(Predict_list), size=12, color='white', horizontalalignment='left',
        #             verticalalignment='top', transform=axs.transAxes)
